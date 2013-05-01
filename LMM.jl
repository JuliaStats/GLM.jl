## Types representing linear mixed models with simple, scalar random effects
## In the LMMsimple type the fixed-effects are incorporated in the
## symmetric, sparse system matrix A.  In the LMMsplit type they are
## separate.

type LMMsimple{Tv<:Float64,Ti<:ITypes} <: SimpleLinearMixedModel
    Zt::SparseMatrixRSC{Tv,Ti}# model matrices Z and X in an RSC structure
    theta::Vector{Tv}          # variance component parameter vector
    lower::Vector{Tv}          # lower bounds (always zeros(length(theta)) for these models)
    A::CholmodSparse{Tv,Ti}    # sparse symmetric system matrix
    anzv::Vector{Tv}           # cached copy of nonzeros from Zt*Zt'
    L::CholmodFactor{Tv,Ti}    # factor of current system matrix
    ubeta::Vector{Tv}          # coefficient vector
    sqrtwts::Vector{Tv}        # square root of weights - can be length 0
    y::Vector{Tv}              # response vector
    Zty::Vector{Tv}           # cached copy of Zt*y
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

type LMMsplit{Tv<:Float64,Ti<:ITypes} <: SimpleLinearMixedModel
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
