module RSCpred

function mktheta(nc)
    mapreduce(j->mapreduce(k->float([1.,zeros(j-k)]), vcat, 1:j), vcat, nc)
end

type RSCpred{Tv<:VTypes,Ti<:ITypes} <: LinPred
    ZXt::SparseMatrixRSC{Tv,Ti}
    theta::Vector{Tv}
    lower::Vector{Tv}
    A::CholmodSparse{Tv,Ti}
    L::CholmodFactor{Tv,Ti}
    ubeta0::Vector{Tv}
    delubeta::Vector{Tv}
    xwts::Vector{Tv}
    avals::Vector{Tv}
    simple::Bool
end

function RSCpred{Tv,Ti}(ZXt::SparseMatrixRSC{Tv,Ti}, theta::Vector)
    aat = A_mul_Bc(ZXt,ZXt) # i.e. ZXt * ZXt' (single quotes confuse syntax highlighting)
    avals = copy(aat.nzval)
    cp = aat.colptr0
    rv = aat.rowval0
    xv = aat.nzval
    for j in 1:ZXt.q                    # inflate the diagonal
        k = cp[j+1]                     # 0-based indices in cp
        assert(rv[k] == j-1)
        xv[k] += 1.
    end
    th  = convert(Vector{Tv},theta)
    ff  = sum(th .== one(Tv))
    if ff != size(ZXt.rowval, 1)
        error("number of finite elements of lower = $ff should be $(size(ZXt.rowval, 1))")
    end
    ub = zeros(Tv,(size(ZXt,1),))
    RSCpred{Tv,Ti}(ZXt, th, [convert(Tv,t == 0.?-Inf:0.) for t in th],
                   aat, cholfact(aat), zeros(Tv,size(ZXt,1)),
                   zeros(Tv,size(ZXt,1)), ones(Tv, size(ZXt,2)), avals, ff == length(theta))
end

function apply_lambda!{T}(vv::Vector{T}, x::RSCpred{T}, wt::T)
    dpos = 0
    low  = x.lower
    th   = x.theta
    off  = 0
    for k in 1:length(low)
        if low[k] == 0.                 # diagonal element of factor
            dpos += 1
            vv[dpos] *= th[k]
            off = 0
        else
            off += 1
            vv[dpos] += th[k] * vv[dpos + off]
        end
    end
    if (wt != 1.) vv *= wt end
    vv
end
    
function updateAL!{Tv,Ti}(x::RSCpred{Tv,Ti})
    if any(x.theta .< x.lower) error("theta violates lower bound") end
    n = size(x.ZXt,2)
    if (length(x.xwts) != n) error("length(xwts) = $(length(x.xwts)) should be $n") end
    ## change this to work with the 0-based indices
    cp  = increment(x.A.colptr0)        # 1-based column pointers and rowvals for A 
    rv  = increment(x.A.rowval0)
    nzv = x.A.nzval
    q   = x.ZXt.q         # number of rows and columns in Zt part of A
    ## Initialize A to the q by q identity in the upper left hand corner,
    ## zeros elsewhere.
    for j in 1:(x.A.c.n), kk in cp[j]:(cp[j+1] - 1)
        nzv[kk] = (rv[kk] == j && j <= q) ? 1. : 0.
    end
    ZXr = x.ZXt.rowval
    ZXv = x.ZXt.nzval
    k   = size(ZXr, 1) # number of non-zeros per column of the Zt part
    w = Array(Tv, size(ZXv, 1))     # avoid reallocation of work array
    for j in 1:n
        w[:] = ZXv[:,j]
        apply_lambda!(w, x, x.xwts[j])
        ## scan up the j'th column of ZXt
        for i in length(w):-1:1
            ii = i <= k ? ZXr[i,j] : q + i - k
            cpi = cp[ii]                # 1-based column pointer
            ll = cp[ii + 1] - 1         # location of diagonal
            nzv[ll] += square(w[i])
            for l in (i-1):-1:1         # off-diagonals
                if ll < cpi break end
                ii1 = l <= k ? ZXr[l,j] : q + l - k
                while (rv[ll] > ii1) ll -= 1 end
                if rv[ll] != ii1 error("Pattern mismatch") end
                nzv[ll] += w[i] * w[l]
            end
        end
    end
    Base.LinAlg.CHOLMOD.chm_factorize!(x.L, x.A)
end

function installubeta(p::RSCpred, f::Real)
    p.ubeta0 += f * p.delubeta
    p.delubeta[:] = zeros(length(p.delubeta))
    p.ubeta0
end
installubeta(p::RSCpred) = installubeta(p, 1.0)
linpred(p::RSCpred, f::Real) = Ac_mul_B(p.ZXt, p.ubeta0 + f * p.delubeta)
linpred(p::RSCpred) = linpred(p, 1.0)
function delubeta{T<:VTypes}(pp::RSCpred{T}, r::Vector{T}) # come up with a better name
    ## blockdiag(Lambdat,I) * ZXt * (p.wts .* r)
    pr = zeros(T, size(pp.ZXt,1))
    rv = pp.ZXt.rowval
    k,n = size(rv)
    p = pp.ZXt.p
    q = pp.ZXt.q
    w = Array(T, k + p)
    for j in 1:n
        rj = r[j]
        w[:] = pp.ZXt.nzval[:,j]
        apply_lambda!(w, pp, pp.xwts[j])
        for i in 1:k pr[rv[i,j]] += w[i] * rj end
        for i in 1:p pr[q+i] += w[k + 1] * rj end
    end
    pp.delubeta[:] = (pp.L \ pr).mat
end

function deviance{T}(pp::RSCpred{T}, resp::Vector{T})
    delubeta(pp, resp)
    p = pp.ZXt.p
    q = pp.ZXt.q
    rss = (s = 0.; for r in (resp - linpred(pp)) s += r*r end; s)
    ldL2 = logdet(pp.L,1:q)
    ldRX2 = logdet(pp.L,(1:p)+q)
    sqrL = (s = 0.; for j in 1:q s += square(pp.delubeta[j]) end; s)
    lnum = log(2pi * (rss + sqrL))
    n = float(size(pp.ZXt,2))
    nmp = n - p
    ldL2 + n * (1 + lnum - log(n)), ldL2 + ldRX2 + nmp * (1 + lnum - log(nmp))
end

function quickupdate{T}(pp::RSCpred{T}, theta::Vector{T})
    ZXt = pp.ZXt
    sc = ones(size(ZXt,1))
    for j in 1:ZXt.q sc[j] = theta[1] end
    A = copy(pp.A)
    xv = A.nzval
    xv[:] = pp.avals[:]
    cp = A.colptr0
    rv = A.rowval0
    Base.LinAlg.CHOLMOD.chm_scale!(A, CholmodDense(sc), 3)
    for j in 1:ZXt.q                    # inflate the diagonal
        k = cp[j+1]                     # 0-based indices in cp
        assert(rv[k] == j-1)
        xv[k] += 1.
    end
    Base.LinAlg.CHOLMOD.chm_factorize!(pp.L, A)
end

end
