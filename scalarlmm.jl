## Types representing linear mixed models with simple, scalar random effects
## The ScalarLMM1 type represents a model with a single scalar
## random-effects term

## Fields are arranged by decreasing size, doubles then pointers then bools
type ScalarLMM1{Ti<:Integer} <: ScalarLinearMixedModel
    theta::Float64
    Xt::Matrix{Float64}
    XtX::Matrix{Float64}
    XtZ::Matrix{Float64}
    Xty::Vector{Float64}
    Ztrv::Vector{Ti}           # indices into factor levels
    Ztnz::Vector{Float64}      # left-hand side of r.e. term
    ZtZ::Vector{Float64}
    Zty::Vector{Float64}
    beta::Vector{Float64}
    mu::Vector{Float64}
    u::Vector{Float64}
    y::Vector{Float64}
    REML::Bool
    fit::Bool
end

function ScalarLMM1{Ti<:Integer}(X::Matrix{Float64}, Ztrv::Vector{Ti},
                                 Ztnz::Vector{Float64}, y::Vector{Float64})
    n,p = size(X)
    if length(Ztrv) != n || length(Ztnz) != n || length(y) != n
        error(string("Dimension mismatch: n = $n, ",
                     "lengths of Ztrv = $(length(Ztrv)), ",
                     "Ztnz = $(length(Ztnz)), y = $(length(y))"))
    end
    q   = length(unique(Ztrv))
    if any(Ztrv .< 1) || any(Ztrv .> q)
        error("All elements of Ztrv must be in 1:$q")
    end
    Xt = X'
    ZtZ = zeros(q); XtZ = zeros(p,q); Zty = zeros(q);
    for i in 1:n
        j = Ztrv[i]; z = Ztnz[i]
        ZtZ[j] += z*z; Zty[j] += z*y[i]; XtZ[:,j] += z*Xt[:,i]
    end
    ScalarLMM1{Ti}(1., Xt, syrk!('L','N',1.,Xt,0.,zeros(p,p)), # XtX
                   XtZ, Xt*y, copy(Ztrv), copy(Ztnz), ZtZ, Zty,
                   zeros(p), zeros(n), zeros(q), copy(y), false, false)
end

function ScalarLMM1{Ti<:Integer}(X::Matrix{Float64}, Ztrv::Vector{Ti},
                                 y::Vector{Float64})
    ScalarLMM1(X, Ztrv, ones(length(Ztrv)), y)
end

lower!(m::ScalarLMM1) = zeros(1)
size(m::ScalarLMM1) = (length(m.y), length(m.beta), length(m.u), 1)
thvec!(m::ScalarLMM1) = [m.theta]
ranef!(m::ScalarLMM1) = m.theta * m.u
grplevels(m::ScalarLMM1) = [length(u)]
Zt!(m::ScalarLMM1) = SparseMatrixRSC(Ztrv, Ztnz')
sqrtwts!(m::ScalarLMM1) = Float64[]
function L(m::ScalarLMM1) # Can probably skip all the Diagonal stuff
    thsq = square(m.theta)
    convert(Vector{Diagonal{Float64}},
            {Diagonal([sqrt(thsq*z + 1.) for z in m.ZtZ.diag])})
end
function objective!(m::ScalarLMM1, th::Float64)
    if th < 0. error("theta = $th must be >= 0") end
    m.theta = th; n,p,q,t = size(m)
    L = [sqrt(th*th*z + 1.)::Float64 for z in m.ZtZ]
    thlinv = th / L
    LXZ = scale(m.XtZ, thlinv)
    LX = cholfact!(syrk!('L', 'N', -1., LXZ, 1., copy(m.XtX)), :L)
    m.u[:] = m.Zty .* thlinv      # initialize u to cu
    m.beta[:] = m.Xty             # initialize beta to Xty
    gemv!('N', -1., LXZ, m.u, 1., m.beta) # cbeta = Xty - RZX'cu
    potrs!('L', LX.UL, m.beta)    # solve for beta in place
    gemv!('T', -1., LXZ, m.beta, 1., m.u) # cu -= RZX'beta
    m.u ./= L                     # solve for u in place
    m.mu[:] = th*(m.u[m.Ztrv].*m.Ztnz) # mu = Z*Lambda*u
    gemv!('T',1.,m.Xt,m.beta,1.,m.mu) # mu += X'beta
    fn = float64(n - (m.REML ? p : 0))
    ldL2 = (s=0.; for l in L s+=log(l) end;2.s)
    obj = ldL2 + fn * (1. + log(2.pi * pwrss(m)/fn))
    if m.REML obj += logdet(RX) end
    obj
end

objective!(m::ScalarLMM1, tv::Vector{Float64}) = objective!(m, tv[1])

objective(m::ScalarLMM1) = objective(m, m.theta)

type NestedZtZ{Ti<:Integer}
    diags::Vector{Vector{Float64}}
    odrv::Vector{Vector{Ti}}
    odnz::Vector{Vector{Float64}}
end

function NestedZtZ{Ti<:Integer}(Ztrv::Matrix{Ti},Ztnz::Matrix{Float64})
    t,n = size(Ztrv) # t is #levels, n is #objs
    if t < 2 error("Use ScalarLMM1 for a single, scalar term") end
    if (t,n) != size(Ztnz)
        error("size(Ztrv) = $(size(Ztrv)) != size(Ztnz) = $(size(Ztnz))")
    end
    uv = [unique(Ztrv[i,:]) for i in 1:t]
    if !all(map(isperm, uv))
        error("unique(Ztrv[l,:]) must be a permutation for all l")
    end
    nl = [length(uu) for uu in uv]
    diags = [zeros(n)::Vector{Float64} for n in nl]
    nod = (t * (t - 1)) >> 1
    odnz = Array(Vector{Float64}, nod)
    odrv = Array(Vector{Ti}, nod)
    pos = 0
    for j in 1:t       # initialize off-diagonals and check nestedness
        nj = nl[j]
        for i in (j+1):t
            pos += 1
            odnz[pos] = zeros(nj)
            rv = zeros(Ti, nj)
            for k in 1:n
                jj = Ztrv[j,k]; ii = Ztrv[i,k]
                if rv[jj] == 0
                    rv[jj] = ii
                else
                    if rv[jj] != ii error("Non-nested factors") end
                end
            end
            odrv[pos] = rv
        end
    end
    for k in 1:n
        pos = 0
        for j in 1:t
            jj = Ztrv[j,k]; zj = Ztnz[j,k]
            diags[j][jj] += zj * zj
            for i in (j+1):t 
                pos += 1
                odnz[pos][jj] += zj * Ztnz[i,k]
            end
        end
    end
    NestedZtZ{Ti}(diags, odrv, odnz)
end
## The ScalarLMMn type represents a model with a multiple scalar
## random-effects terms that have nested grouping factors.

type ScalarLMMnest{Ti<:Integer} <: ScalarLinearMixedModel
    theta::Vector{Float64}
    Xt::Matrix{Float64}
    XtX::Matrix{Float64}
    XtZ::Vector{Matrix{Float64}}
    Xty::Vector{Float64}
    Ztrv::Matrix{Ti}           # indices into factor levels
    Ztnz::Matrix{Float64}      # left-hand side of r.e. term
    ZtZ::NestedZtZ{Ti}
    Zty::Vector{Vector{Float64}}
    beta::Vector{Float64}
    mu::Vector{Float64}
    u::Vector{Float64}
    y::Vector{Float64}
    REML::Bool
    fit::Bool
end

function ScalarLMMnest{Ti<:Integer}(Xt::Matrix{Float64}, Ztrv::Matrix{Ti},
                                    Ztnz::Matrix{Float64}, y::Vector{Float64})
    p,n = size(Xt); t = size(Ztrv,1)
    ZtZ = NestedZtZ(Ztrv, Ztnz)
    if size(Ztrv,2) != n || length(y) != n 
        error(string("Dimension mismatch: n = $n, ",
                     "sizes of Ztrv = $(size(Ztrv)), ",
                     "Ztnz = $(size(Ztnz)), y = $(size(y))"))
    end
    nl = [length(d)::Int for d in ZtZ.diags]
    q  = sum(nl)
    Zty = [zeros(l)::Vector{Float64} for l in nl]
    XtZ = [zeros(p,l)::Matrix{Float64} for l in nl]
    for i in 1:n, l in 1:t
        j = Ztrv[l,i]; z = Ztnz[l,i]
        Zty[l][j] += z*y[i]; XtZ[l][:,j] += z*Xt[:,i]
    end
    ScalarLMMnest{Ti}(ones(t), Xt, syrk!('L','N',1.,Xt,0.,zeros(p,p)), # XtX
                   XtZ, Xt*y, copy(Ztrv), copy(Ztnz), ZtZ, Zty,
                   zeros(p), zeros(n), zeros(q), copy(y), false, false)
end

function ScalarLMMnest{Ti<:Integer}(X::Matrix{Float64}, Ztrv::Matrix{Ti},
                                 y::Vector{Float64})
    ScalarLMM1(X, Ztrv, ones(size(Ztrv)), y)
end


## In the LMMsimple type the fixed-effects are incorporated in the
## symmetric, sparse system matrix A.  In the LMMsplit type they are
## separate.

type LMMsimple{Tv<:Float64,Ti<:ITypes} <: ScalarLinearMixedModel
    Zt::SparseMatrixRSC{Tv,Ti}# model matrices Z and X in an RSC structure
    theta::Vector{Tv}          # variance component parameter vector
    lower::Vector{Tv}          # lower bounds (always zeros(length(theta)) for these models)
    A::CholmodSparse{Tv,Ti}    # sparse symmetric system matrix
    anzv::Vector{Tv}           # cached copy of nonzeros from Zt*Zt'
    L::CholmodFactor{Tv,Ti}    # factor of current system matrix
    ubeta::Vector{Tv}          # coefficient vector
    sqrtwts::Vector{Tv}        # square root of weights - can be length 0
    y::Vector{Tv}              # response vector
    Zty::Vector{Tv}            # cached copy of Zt*y
    lambda::Vector{Tv}         # diagonal of Lambda
    mu::Vector{Tv}             # fitted response vector
    REML::Bool                 # should a reml fit be used?
    fit::Bool                  # has the model been fit?
end

function LMMsimple{Tv<:Float64,Ti<:ITypes}(Zt::SparseMatrixRSC{Tv,Ti},
                                           y::Vector{Tv},wts::Vector{Tv})
    n = size(Zt,2)
    if length(y) != n error("size(Zt,2) = $(size(Zt,2)) and length(y) = $length(y)") end
    lwts = length(wts)
    if lwts != 0 && lwts != n error("length(wts) = $lwts should be 0 or length(y) = $n") end
    A = Zt*Zt'
    anzv = copy(A.nzval)
    pluseye!(A, 1:Zt.q)
    L = cholfact(A)
    Zty = Zt*y
    ubeta = vec((L\Zty))
    k = size(Zt.rowval, 1)
    LMMsimple{Tv,Ti}(Zt, ones(k), zeros(k), A, anzv, L,
                     ubeta, sqrt(wts), y, Zty, ones(size(L,1)), Ac_mul_B(Zt,ubeta),
                     false, false)
end
LMMsimple(Zt,y) = LMMsimple(Zt, y, Array(eltype(y), 0))

function LMMsimple{Tv<:Float64,Ti<:ITypes}(inds::Matrix{Ti},
                                           X::Matrix{Tv},
                                           y::Vector{Tv},
                                           wts::Vector{Tv})
    n = length(y)
    if !(size(inds,1) == size(X,1) == n) error("Dimension mismatch") end
    ii = copy(inds)
    for j in 2:size(ii,2) ii[:,j] += max(ii[:,j-1]) end
    LMMsimple(SparseMatrixRSC(ii', [ones(size(ii)) X]'), y, wts)
end
function LMMsimple{Tv<:Float64,Ti<:ITypes}(inds::Matrix{Ti}, X::Matrix{Tv}, y::Vector{Tv})
    LMMsimple(inds, X, y, Array(Tv, 0))
end

function LMMsimple(f::Formula, df::AbstractDataFrame)
    mf = ModelFrame(f, df)
    mm = ModelMatrix(mf)
    re = retrms(mf)
    if length(re) == 0 error("No random-effects terms were specified") end
    if !issimple(re) error("only simple random-effects terms allowed") end 
    LMMsimple(grpfac(re,mf), mm.m, dv(model_response(mf)))
end
LMMsimple(ex::Expr, df::AbstractDataFrame) = LMMsimple(Formula(ex), df)

ussq(m::LMMsimple) = (s = 0.; for i in 1:m.Zt.q s += square(m.ubeta[i]) end; s)

size(m::LMMsimple) = (t = m.Zt; (size(t, 2), t.p, t.q))

## Update L, solve for ubeta and evaluate mu
function updatemu(m::LMMsimple)
    m.A.nzval[:] = m.anzv[:]            # restore A to ZXt*ZXt', update A and L
    chm_factorize!(m.L, pluseye!(chm_scale!(m.A, m.lambda, 3), 1:m.Zt.q))
    m.ubeta[:] = (m.L \ (m.lambda .* m.Zty))[:]
    m.mu[:] = m.Zt'*(m.lambda .* m.ubeta)
end

logdetLRX(m::LMMsimple) = logdet(m.L)
logdetL(m::LMMsimple) = logdet(m.L, 1:size(m)[3])

fixef(m::LMMsimple) =  m.ubeta[m.Zt.q + (1:m.Zt.p)]

ranef(m::LMMsimple) = (m.lambda .* m.ubeta)[1:m.Zt.q]

type LMMsplit{Tv<:Float64,Ti<:ITypes} <: ScalarLinearMixedModel
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
    u::Vector{Tv}              # spherical random effects
    beta::Vector{Tv}           # fixed-effects coefficient vector
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
    A = Zt*Zt'
    anzv = copy(A.nzval)
    L = cholfact(pluseye!(A), true)     # force an LL' factorization
    P = (Ti == Int) ? increment(L.Perm) : increment!(int(L.Perm))
    XtX = X'X
    RX = cholfact(XtX)
    k = size(Zt.rowval, 1)
    LMMsplit{Tv,Ti}(Zt, X, ones(k), zeros(k), A, anzv, L, P, Zt*X, Zt*y, XtX, X'y,
                    RX, zeros(L.c.n), zeros(size(X,2)), sqrt(wts), y, zeros(L.c.n),
                    zeros(size(X,1)), false, false)
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
    re = retrms(mf)
    if length(re) == 0 error("No random-effects terms were specified") end
    if !issimple(re) error("only simple random-effects terms allowed") end
    LMMsplit(grpfac(re,mf), mm.m, dv(model_response(mf)))
end
LMMsplit(ex::Expr, df::AbstractDataFrame) = LMMsplit(Formula(ex), df)

## Update L, solve for ubeta and evaluate mu
function updatemu(m::LMMsplit)
    m.A.nzval[:] = m.anzv[:]            # restore A to ZXt*ZXt', update A and L
    chm_factorize_p!(m.L, chm_scale!(m.A, m.lambda, 3), 1.)
    ## solve for RZX and cu
    RZX = solve(m.L, scale(m.lambda, m.ZtX)[m.P,:], CHOLMOD_L)
    cu = solve(m.L, (m.lambda .* m.Zty)[m.P], CHOLMOD_L)
    ## update m.RX
    m.RX.UL[:] = m.XtX[:]
    _, info = potrf!(m.RX.uplo, syrk!(m.RX.uplo, 'T', -1., RZX, 1., m.RX.UL))
    if info != 0
        error("Rank of downdated X'X is $info at theta = $(m.theta)")
    end
    m.beta[:] = m.RX\(m.Xty - RZX'cu)
    m.u[m.P] = solve(m.L, cu - RZX*m.beta, CHOLMOD_Lt)
    m.mu[:] = m.X*m.beta + m.Zt'*(m.lambda .* m.u)
end

logdetLRX(m::LMMsplit) = logdet(m.L) + logdet(m.RX)

logdetL(m::LMMsplit) = logdet(m.L)

ussq(m::LMMsplit) = (s = 0.; for u in m.u s += u*u end; s)

## Should think of a clever way of creating a 3-tuple from size(m.X) and length(m.u)
size(m::LMMsplit) = (size(m.X, 1), length(m.beta), length(m.u))

fixef(m::LMMsplit) = m.beta

ranef(m::LMMsplit) = m.lambda .* m.u

rowvalZt(m::LMMsplit) = m.Zt.rowval
