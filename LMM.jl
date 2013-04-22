type LMMsimple{Tv<:Float64,Ti<:ITypes} <: LinearMixedModel
    ZXt::SparseMatrixRSC{Tv,Ti}# model matrices Z and X in an RSC structure
    theta::Vector{Tv}          # variance component parameter vector
    lower::Vector{Tv}          # lower bounds (always zeros(length(theta)) for these models)
    A::CholmodSparse{Tv,Ti}    # sparse symmetric system matrix
    anzv::Vector{Tv}           # cached copy of nonzeros from ZXt*ZXt'
    L::CholmodFactor{Tv,Ti}    # factor of current system matrix
    ubeta::Vector{Tv}          # coefficient vector
    sqrtwts::Vector{Tv}        # square root of weights - can be length 0
    y::Vector{Tv}              # response vector
    ZXty::Vector{Tv}           # cached copy of ZXt*y
    lambda::Vector{Tv}         # diagonal of Lambda
    mu::Vector{Tv}             # fitted response vector
    REML::Bool                 # should a reml fit be used?
    fit::Bool                  # has the model been fit?
end

## Add an identity block along inds to a symmetric A stored in the upper triangle
function pluseye!{T}(A::CholmodSparse{T}, inds)
    if A.c.stype <= 0 error("Matrix A must be symmetric and stored in upper triangle") end
    cp = A.colptr0
    rv = A.rowval0
    xv = A.nzval
    for j in inds
        k = cp[j+1]
        assert(rv[k] == j-1)
        xv[k] += one(T)
    end
    A
end
pluseye!(A::CholmodSparse) = pluseye!(A,1:size(A,1))

function LMMsimple{Tv<:Float64,Ti<:ITypes}(ZXt::SparseMatrixRSC{Tv,Ti},
                                           y::Vector{Tv},wts::Vector{Tv})
    n = size(ZXt,2)
    if length(y) != n error("size(ZXt,2) = $(size(ZXt,2)) and length(y) = $length(y)") end
    lwts = length(wts)
    if lwts != 0 && lwts != n error("length(wts) = $lwts should be 0 or length(y) = $n") end
    A = ZXt*ZXt'
    anzv = copy(A.nzval)
    pluseye!(A, 1:ZXt.q)
    L = cholfact(A)
    ZXty = ZXt*y
    ubeta = vec((L\ZXty))
    k = size(ZXt.rowval, 1)
    LMMsimple{Tv,Ti}(ZXt, ones(k), zeros(k), A, anzv, L,
                     ubeta, sqrt(wts), y, ZXty, ones(size(L,1)), Ac_mul_B(ZXt,ubeta),
                     false, false)
end
LMMsimple(ZXt,y) = LMMsimple(ZXt, y, Array(eltype(y), 0))

function LMMsimple{Tv<:Float64,Ti<:ITypes}(inds::Matrix{Ti},
                                           X::Matrix{Tv},
                                           y::Vector{Tv},
                                           wts::Vector{Tv})
    n = length(y)
    if !(size(inds,1) == size(X,1) == n) error("Dimension mismatch") end
    ii = droplevels(inds)
    for j in 2:size(ii,2) ii[:,j] += max(ii[:,j-1]) end
    LMMsimple(SparseMatrixRSC(ii', [ones(size(ii)) X]'), y, wts)
end
function LMMsimple{Tv<:Float64,Ti<:ITypes}(inds::Matrix{Ti}, X::Matrix{Tv}, y::Vector{Tv})
    LMMsimple(inds, X, y, Array(Tv, 0))
end

dv(da::DataArray) = da.data
dv{T<:Number}(vv::Vector{T}) = vv

function LMMsimple(f::Formula, df::AbstractDataFrame)
    mf = ModelFrame(f, df)
    mm = ModelMatrix(mf)
    re = filter(x->Meta.isexpr(x,:call) && x.args[1] == :|, mf.terms.terms)
    if length(re) == 0 error("No random-effects terms were specified") end
    simple = map(x->x.args[2] == 1, re)
    if !all(simple) error("only simple random-effects terms allowed") end
    inds = int32(hcat(map(x->mf.df[x.args[3]].refs,re)...)) # use a droplevels here
    LMMsimple(inds, mm.m, dv(model_response(mf)))
end
LMMsimple(ex::Expr, df::AbstractDataFrame) = LMMsimple(Formula(ex), df)

function droplevels{T<:ITypes}(inds::Matrix{T})
    ic = copy(inds)
    m,n = size(ic)
    for j in 1:n
        uj = unique(ic[:,j])
        nuj = length(uj)
        if min(uj) == 1 && max(uj) == nuj break end
        suj = sort!(uj)
        dict = Dict(suj, one(T):convert(T,nuj))
        ic[:,j] = [ dict[ic[i,j]] for i in 1:m ]
    end
    ic
end

function fill!{T}(a::Vector{T}, x::T, inds)
    for i in inds a[i] = x end
end

function deviance(m::LMMsimple,theta::Vector{Float64})
    if length(theta) != length(m.theta) error("Dimension mismatch") end
    if any(theta .< 0.) error("all elements of theta must be non-negative") end
    m.theta[:] = theta[:]               # copy in place, update lambda vector
    for i in 1:length(theta)
        fill!(m.lambda, theta[i], int(m.ZXt.rowrange[i]))
    end
    m.A.nzval[:] = m.anzv[:]            # restore A to ZXt*ZXt', update A and L
    chm_factorize!(m.L, pluseye!(chm_scale!(m.A, m.lambda, 3), 1:m.ZXt.q))
                                        # solve for ubeta and evaluate mu
    m.ubeta[:] = (m.L \ (m.lambda .* m.ZXty))[:]
    m.mu[:] = m.ZXt'*(m.lambda .* m.ubeta)
    rss = (s = 0.; for r in (m.y - m.mu) s += r*r end; s)
    ussq = (s = 0.; for j in 1:m.ZXt.q s += square(m.ubeta[j]) end; s)
    lnum = log(2pi * (rss+ussq))
    if m.REML
        nmp = float(size(m.ZXt,2) - m.ZXt.p)
        return logdet(m.L) + nmp * (1 + lnum - log(nmp))
    end
    n = float(size(m.ZXt,2))
    return logdet(m.L, 1:m.ZXt.q) + n * (1 + lnum - log(n))
end

function fit(m::LinearMixedModel, verbose::Bool)
    if !m.fit
        k = length(m.theta)
        opt = Opt(:LN_BOBYQA, k)
        ftol_abs!(opt, 1e-6)    # criterion on deviance changes
        xtol_abs!(opt, 1e-6)    # criterion on all parameter value changes
        lower_bounds!(opt, m.lower)
        function obj(x::Vector{Float64}, g::Vector{Float64})
            if length(g) > 0 error("gradient evaluations are not provided") end
            deviance(m, x)
        end
        if verbose
            count = 0
            function vobj(x::Vector{Float64}, g::Vector{Float64})
                if length(g) > 0 error("gradient evaluations are not provided") end
                count += 1
                val = obj(x, g)
                println("f_$count: $val, $x")
                val
            end
            min_objective!(opt, vobj)
        else
            min_objective!(opt, obj)
        end
        fmin, xmin, ret = optimize(opt, m.theta)
        if verbose println(ret) end
        m.fit = true
    end
    m
end
fit(m::LinearMixedModel) = fit(m, false)      # non-verbose

deviance(m::LMMsimple) = (fit(m); deviance(m,m.theta))

function fixef(m::LMMsimple)
    fit(m)
    m.ubeta[m.ZXt.q + (1:m.ZXt.p)]
end

function ranef(m::LMMsimple)
    fit(m)
    (m.lambda .* m.ubeta)[1:m.ZXt.q]
end

function VarCorr(m::LMMsimple)
    fit(m)
    pwrss = (s = 0.; for r in (m.y - m.mu) s += r*r end; s)
    pwrss += (s = 0.; for i in 1:m.ZXt.q s += square(m.ubeta[i]) end; s)
    vec([m.theta.^2, 1.] * pwrss/(length(m.y) - (m.REML ? m.ZXt.p : 0)))
end

reml(m::LMMsimple) = (m.REML = true; m.fit = false; m)
    
function show(io::IO, m::LMMsimple)
    fit(m)
    REML = m.REML
    criterionstr = REML ? "REML" : "maximum likelihood"
    println(io, "Linear mixed model fit by $criterionstr")
    dd = deviance(m)
    if REML
        println(io, " REML criterion: $dd")
    else
        println(io, " logLik: $(-dd/2), deviance: $dd")
    end
    vc = VarCorr(m)
    println("\n  Variance components: $vc")
    ZXt = m.ZXt
    rv = ZXt.rowval
    grplevs = [length(unique(rv[i,:]))::Int for i in 1:size(rv,1)]
    println("  Number of obs: $(length(m.y)); levels of grouping factors: $grplevs")
    println("  Fixed-effects parameters: $(fixef(m))")
end

type LMMsplit{Tv<:Float64,Ti<:ITypes} <: LinearMixedModel
    Zt::SparseMatrixRSC{Tv,Ti} # random-effects model matrix Z
    X::Matrix{Tv}              # fixed-effects model matrix
    theta::Vector{Tv}          # variance component parameter vector
    lower::Vector{Tv}          # lower bounds (always zeros(length(theta)) for these models)
    A::CholmodSparse{Tv,Ti}    # sparse symmetric random-effects system matrix
    anzv::Vector{Tv}           # cached copy of nonzeros from Zt*Zt'
    L::CholmodFactor{Tv,Ti}    # factor of current system matrix
    P::Vector{Int}             # permutation vector for L
    ZtX::Matrix{Tv}            # cached copy of Zt*X
    Zty::Vector{Tv}            # cached copy of Zt*y
    XtX::Matrix{Tv}            # cached copy of X'X
    Xty::Vector{Tv}            # cached copy of X'y
    RX::Cholesky{Tv}           # fixed-effects part of the Cholesky factor
    ubeta::Vector{Tv}          # coefficient vector
    sqrtwts::Vector{Tv}        # square root of weights - can be length 0
    y::Vector{Tv}              # response vector
    lambda::Vector{Tv}         # diagonal of Lambda
    mu::Vector{Tv}             # fitted response vector
    REML::Bool                 # should a reml fit be used?
    fit::Bool                  # has the model been fit?
end

function LMMsplit{Tv<:Float64,Ti<:ITypes}(Zt::SparseMatrixRSC{Tv,Ti},
                                          X::Matrix{Tv},
                                          y::Vector{Tv},
                                          wts::Vector{Tv})
    n = size(X,1)
    if length(y) != n || size(Zt,2) != n
        error("size(Zt,2) = $(size(Zt,2)) and length(y) = $length(y) and size(X) = $(size(X))")
    end
    lwts = length(wts)
    if lwts != 0 && lwts != n error("length(wts) = $lwts should be 0 or length(y) = $n") end
    A = A_mul_Bc(Zt,Zt) # i.e. Zt*Zt' (single quotes confuse syntax highlighting)
    anzv = copy(A.nzval)
    L = cholfact(pluseye!(A), true)     # force an LL' factorization
    P = (Ti == Int) ? increment(L.Perm) : increment!(int(L.Perm))
    XtX = X'X
    RX = cholfact(XtX)
    Xty = X'y
    Zty = Zt*y
    ubeta = vcat(vec(L\Zty), RX\Xty)
    k = size(Zt.rowval, 1)
    LMMsplit{Tv,Ti}(Zt, X, ones(k), zeros(k), A, anzv, L, P,
                    Zt*X, Zty, XtX, Xty, RX,
                     ubeta, sqrt(wts), y, ones(size(L,1)), Zt'*ubeta[1:Zt.q], #'
                     false, false)
end

function LMMsplit{Tv<:Float64,Ti<:ITypes}(inds::Matrix{Ti}, X::Matrix{Tv}, y::Vector{Tv})
    LMMsplit(inds, X, y, Array(Tv, 0))
end
function LMMsplit{Tv<:Float64,Ti<:ITypes}(inds::Matrix{Ti},
                                          X::Matrix{Tv},
                                          y::Vector{Tv},
                                          wts::Vector{Tv})
    n = length(y)
    if !(size(inds,1) == size(X,1) == n) error("Dimension mismatch") end
    ii = copy(inds)
    for j in 2:size(ii,2) ii[:,j] += max(ii[:,j-1]) end
    LMMsplit(SparseMatrixRSC(ii', ones(size(ii))'), X, y, wts)
end
function LMMsplit(f::Formula, df::AbstractDataFrame)
    mf = ModelFrame(f, df)
    mm = ModelMatrix(mf)
    re = filter(x->Meta.isexpr(x,:call) && x.args[1] == :|, mf.terms.terms)
    if length(re) == 0 error("No random-effects terms were specified") end
    simple = map(x->x.args[2] == 1, re)
    if !all(simple) error("only simple random-effects terms allowed") end
    inds = int32(hcat(map(x->mf.df[x.args[3]].refs,re)...)) # use a droplevels here
    LMMsplit(inds, mm.m, dv(model_response(mf)))
end
LMMsplit(ex::Expr, df::AbstractDataFrame) = LMMsplit(Formula(ex), df)

using Base.LinAlg.BLAS.syrk!
using Base.LinAlg.LAPACK.potrf!
using Base.LinAlg.CHOLMOD.CHOLMOD_P, Base.LinAlg.CHOLMOD.CHOLMOD_L
using Base.LinAlg.CHOLMOD.CHOLMOD_Pt, Base.LinAlg.CHOLMOD.CHOLMOD_Lt
function deviance(m::LMMsplit,theta::Vector{Float64})
    if length(theta) != length(m.theta) error("Dimension mismatch") end
    if any(theta .< m.lower)
        error("theta = $theta violates lower bounds $(m.lower)")
    end
    m.theta[:] = theta[:]               # copy in place
    m.A.nzval[:] = m.anzv[:]            # restore A in place to Zt*Zt'
    for i in 1:length(theta)            # update Lambda (stored as a vector)
        fill!(m.lambda, theta[i], int(m.Zt.rowrange[i]))
    end
    ## scale Z'Z to Lambda'Z'Z*Lambda, add I and update m.L
    chm_factorize!(m.L, pluseye!(chm_scale!(m.A, m.lambda, 3)))
    ## solve for RZX and cu
    RZX = solve(m.L, diagmm(m.lambda, m.ZtX)[m.P,:], CHOLMOD_L)
    cu = solve(m.L, (m.lambda .* m.Zty)[m.P], CHOLMOD_L)
    ## update m.RX
    m.RX.UL[:] = m.XtX[:]
    _, info = potrf!(m.RX.uplo, syrk!(m.RX.uplo, 'T', -1., RZX, 1., m.RX.UL))
    if info != 0
        error("Rank of downdated X'X is $info at theta = $theta")
    end
    beta = m.RX\(m.Xty - RZX'cu)
    m.ubeta[m.P] = solve(m.L, cu - RZX*beta, CHOLMOD_Lt)[:]
    m.ubeta[length(m.P) + (1:length(beta))] = beta
    m.mu[:] = m.X*beta + m.Zt'*(m.lambda .* (m.ubeta[1:length(m.P)]))
    rss = (s = 0.; for r in (m.y - m.mu) s += r*r end; s)
    ussq = (s = 0.; for j in 1:m.Zt.q s += square(m.ubeta[j]) end; s)
    lnum = log(2pi * (rss + ussq))
    n,p = size(m.X)
    if m.REML
        nmp = float(n - p)
        return logdet(m.L) + nmp * (1 + lnum - log(nmp))
    end
    return logdet(m.L) + n * (1 + lnum - log(n))
end
