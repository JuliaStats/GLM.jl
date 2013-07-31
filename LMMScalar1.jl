## Types representing linear mixed models with simple, scalar random effects
## The ScalarLMM1 type represents a model with a single scalar
## random-effects term

abstract LMMScalar <: LinearMixedModel

function VarCorr(m::LMMScalar)
    n, p, q = size(m)
    [m.theta .^ 2, 1.] * (pwrss(m)/float(n - (m.REML ? p : 0)))
end

## Fields are arranged by decreasing size, doubles then pointers then bools
type LMMScalar1{Ti<:Integer} <: LMMScalar
    theta::Float64
    L::Vector{Float64}
    RX::Cholesky{Float64}
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

if false
    function LMMScalar1{Ti<:Integer}(X::Matrix{Float64}, Ztrv::Vector{Ti},
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
        LMMScalar1{Ti}(1., Xt, syrk!('L','N',1.,Xt,0.,zeros(p,p)), # XtX
                       XtZ, Xt*y, copy(Ztrv), copy(Ztnz), ZtZ, Zty,
                       zeros(p), zeros(n), zeros(q), copy(y), false, false)
    end

    function LMMScalar1{Ti<:Integer}(X::Matrix{Float64}, Ztrv::Vector{Ti},
                                     y::Vector{Float64})
        LMMScalar1(X, Ztrv, ones(length(Ztrv)), y)
    end
end

##  cholfact(x, RX=true) -> the Cholesky factor of the downdated X'X or LambdatZt
cholfact(m::LMMScalar,RX=true) = RX ? m.RX : Diagonal(m.L)

grplevels(m::LMMScalar1) = [length(m.u)]

isscalar(m::LMMScalar1) = true

## linpred!(m) -> m   -- update mu
function linpred!(m::LMMScalar1)
    for i in 1:length(m.mu)             # mu = Z*Lambda*u
        m.mu[i] = m.theta * m.u[m.Ztrv[i]] * m.Ztnz[i]
    end
    gemv!('T',1.,m.Xt,m.beta,1.,m.mu)   # mu += X'beta
    m
end

## Logarithm of the determinant of the generator matrix for the Cholesky factor, RX or L
logdet(m::LMMScalar1,RX=true) = RX ? logdet(m.RX) : 2.sum(Log(),m.L)

## lower(m) -> lower bounds on elements of theta
lower(m::LMMScalar1) = zeros(1)

## sqrlenu(m) -> total squared length of m.u (the penalty in the PLS problem)
sqrlenu(m::LMMScalar1) = sqsum(m.u)

##  ranef(m) -> vector of matrices of random effects on the original scale
##  ranef(m,true) -> vector of matrices of random effects on the U scale
function ranef(m::LMMScalar1, uscale=false)
    uscale && return [m.u']
    [m.theta * m.u']
end

##  size(m) -> n, p, q, t (lengths of y, beta, u and # of re terms)
size(m::LMMScalar1) = (length(m.y), length(m.beta), length(m.u), 1)

theta(m::LMMScalar1) = [m.theta]

##  theta!(m,th) -> m : install new value of theta, update LambdatZt and L 
function theta!(m::LMMScalar1, th::Vector{Float64})
    length(th) == 1 || error("LMMScalar1 theta must have length 1")
    m.theta = th[1]; n,p,q,t = size(m)
    m.theta >= 0. || error("theta = $th must be >= 0")
    map!(x->sqrt(m.theta*m.theta*x + 1.), m.L, m.ZtZ)
    m
end

## solve!(m) -> m : solve PLS problem for u given beta
## solve!(m,true) -> m : solve PLS problem for u and beta
function solve!(m::LMMScalar1, ubeta=false)
    thlinv = m.theta / m.L
    map!(Multiply(), m.u, m.Zty, thlinv) # initialize u to cu
    if ubeta
        LXZ = scale(m.XtZ, thlinv)
        potrf!('U', syrk!('U', 'N', -1., LXZ, 1., copy!(m.RX.UL, m.XtX)))
        copy!(m.beta, m.Xty)              # initialize beta to Xty
        gemv!('N',-1.,LXZ,m.u,1.,m.beta)  # cbeta = Xty - RZX'cu
        solve!(m.RX, m.beta)              # solve for beta in place
        gemv!('T',-1.,LXZ,m.beta,1.,m.u)  # cu -= RZX'beta
    end
    map1!(Divide(), m.u, m.L)           # solve for u in place
    linpred!(m)                         # update mu
end

## std(m) -> Vector{Vector{Float64}} estimated standard deviations of variance components
std(m::LMMScalar1) = scale(m)*[m.theta, 1.]
