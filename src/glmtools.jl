const minfloat = realmin(Float64)
const oneMeps  = 1. - eps()
const llmaxabs = log(-log(minfloat))
const logeps   = log(eps())
abstract Link                           # Link types define linkfun, linkinv, mueta,
                                        # valideta and validmu.

chkpositive(x::Real) = isfinite(x) && 0. < x ? x : error("argument must be positive")
chkfinite(x::Real) = isfinite(x) ? x : error("argument must be finite")
clamp01(x::Real) = clamp(x, minfloat, oneMeps)
chk01(x::Real) = 0. < x < 1. ? x : error("argument must be in (0,1)")

type CauchitLink  <: Link end
linkfun(l::CauchitLink, mu::Real) = tan(pi * (mu - 0.5))
linkinv(l::CauchitLink, eta::Real) = 0.5 + atan(eta) / pi
mueta(l::CauchitLink, eta::Real) = 1. /(pi * (1 + eta * eta))
valideta(l::CauchitLink, eta::Real) = chkfinite(eta)
validmu(l::CauchitLink, mu::Real) = chk01(mu)

type CloglogLink  <: Link end
linkfun (l::CloglogLink,   mu::Real) = log(-log(1. - mu))
linkinv (l::CloglogLink,  eta::Real) = -expm1(-exp(eta))
mueta   (l::CloglogLink,  eta::Real) = exp(eta) * exp(-exp(eta))
valideta(l::CloglogLink,  eta::Real) = abs(eta) < llmaxabs? eta: error("require abs(eta) < $llmaxabs")
validmu (l::CloglogLink,   mu::Real) = chk01(mu)

type IdentityLink <: Link end
linkfun (l::IdentityLink,  mu::Real) = mu
linkfun!{T<:FloatingPoint}(l::IdentityLink, eta::Vector{T}, mu::Vector{T}) = copy!(eta, mu)
linkinv (l::IdentityLink, eta::Real) = eta
linkinv!{T<:FloatingPoint}(l::IdentityLink, mu::Vector{T}, eta::Vector{T}) = copy!(mu, eta)         
mueta   (l::IdentityLink, eta::Real) = 1.
mueta!{T<:FloatingPoint}(l::IdentityLink, me::Vector{T}, eta::Vector{T}) = fill!(me, one(T))
mueta{T<:FloatingPoint}(l::IdentityLink, eta::Vector{T}) = ones(T, length(eta))
valideta(l::IdentityLink, eta::Real) = chkfinite(eta)
validmu (l::IdentityLink,  mu::Real) = chkfinite(mu)

type InverseLink  <: Link end
linkfun (l::InverseLink,   mu::Real) =  1. / mu
linkinv (l::InverseLink,  eta::Real) =  1. / eta
mueta   (l::InverseLink,  eta::Real) = -1. / (eta * eta)
valideta(l::InverseLink,  eta::Real) = chkpositive(eta)
validmu (l::InverseLink,  eta::Real) = chkpositive(mu)

type LogitLink    <: Link end
linkfun (l::LogitLink,     mu::Real) = logit(mu)
linkfun!{T<:FloatingPoint}(l::LogitLink, eta::Vector{T}, mu::Vector{T}) = map!(Logit(), eta, mu)
linkinv (l::LogitLink,    eta::Real) = logistic(mu)
linkinv!{T<:FloatingPoint}(l::LogitLink, mu::Vector{T}, eta::Vector{T}) = map!(Logistic(), mu, eta)
mueta   (l::LogitLink,    eta::Real) = (e = exp(-abs(eta)); f = 1. + e; e / (f * f))
type LogistDens <: UnaryFunctor end
evaluate{T<:FloatingPoint}(::LogistDens,x::T) = (e = exp(-abs(x)); f = one(T) + e; e / (f * f))
result_type{T<:FloatingPoint}(::LogistDens, ::Type{T}) = T
mueta!{T<:FloatingPoint}(l::LogitLink, me::Vector{T}, eta::Vector{T}) = map!(LogistDens(), me, eta)
mueta{T<:FloatingPoint}(l::LogitLink, eta::Vector{T}) = map(LogistDens(), eta)
valideta(l::LogitLink,    eta::Real) = chkfinite(eta)
validmu (l::LogitLink,     mu::Real) = chk01(mu)

type LogLink      <: Link end
linkfun (l::LogLink,       mu::Real) = log(mu)
linkfun!{T<:FloatingPoint}(l::LogLink, eta::Vector{T}, mu::Vector{T}) = map!(Log(), eta, mu)
linkinv (l::LogLink,      eta::Real) = exp(eta)
linkinv!{T<:FloatingPoint}(l::LogLink, mu::Vector{T}, eta::Vector{T}) = map!(Exp(), mu, eta)
mueta   (l::LogLink,      eta::Real) = eta < logeps ? eps() : exp(eta)
mueta!{T<:FloatingPoint}(l::LogLink, me::Vector{T}, eta::Vector{T}) = map!(Exp(), mu, eta)
valideta(l::LogLink,      eta::Real) = chkfinite(eta)
validmu (l::LogLink,       mu::Real) = chkpositive(mu)

type ProbitLink   <: Link end
linkfun (l::ProbitLink,    mu::Real) = ccall((:qnorm5, Rmath), Float64,
                                             (Float64,Float64,Float64,Int32,Int32),
                                             mu, 0., 1., 1, 0)
linkinv (l::ProbitLink,   eta::Real) = (1. + erf(eta/sqrt(2.))) / 2.
mueta   (l::ProbitLink,   eta::Real) = exp(-0.5eta^2) / sqrt(2.pi)
valideta(l::ProbitLink,   eta::Real) = chkfinite(eta)
validmu (l::ProbitLink,    mu::Real) = chk01(mu)
                                        # Vectorized methods, including validity checks
function linkfun{T<:Real}(l::Link, mu::AbstractArray{T,1})
    [linkfun(l, validmu(l, m)) for m in mu]
end

function linkinv{T<:Real}(l::Link, eta::AbstractArray{T,1})
    [linkinv(l, valideta(l, et)) for et in eta]
end

function mueta{T<:Real}(l::Link, eta::AbstractArray{T,1})
    [mueta(l, valideta(l, et)) for et in eta]
end

canonicallink(d::Gamma)     = InverseLink()
canonicallink(d::Normal)    = IdentityLink()
canonicallink(d::Bernoulli) = LogitLink()
canonicallink(d::Poisson)   = LogLink()

type BernoulliVar <: UnaryFunctor end
evaluate{T<:FloatingPoint}(::BernoulliVar,x::T) = x * (one(T) - x)
result_type{T<:FloatingPoint}(::BernoulliVar, ::Type{T}) = T

var!{T<:FloatingPoint}(d::Bernoulli,v::Vector{T},mu::Vector{T}) = map!(BernoulliVar(),v,mu)
var!{T<:FloatingPoint}(d::Gamma,v::Vector{T},mu::Vector{T}) = map!(Abs2(),v,mu)
var!{T<:FloatingPoint}(d::Normal,v::Vector{T},mu::Vector{T}) = fill!(v,one(T))
var!{T<:FloatingPoint}(d::Poisson,v::Vector{T},mu::Vector{T}) = copy!(v,mu)

type BernoulliMuStart <: BinaryFunctor end
evaluate{T<:FloatingPoint}(::BernoulliMuStart,y::T,w::T) = (w*y + convert(T,0.5))/(w + one(T))
result_type{T<:FloatingPoint}(::BernoulliMuStart,::Type{T},::Type{T}) = T

mustart{T<:FloatingPoint}(d::Bernoulli,y::Vector{T},wt::Vector{T}) = map(BernoulliMuStart(),y,wt)
mustart{T<:FloatingPoint}(d::Gamma,y::Vector{T},wt::Vector{T}) = copy(y)
mustart{T<:FloatingPoint}(d::Normal,y::Vector{T},wt::Vector{T}) = copy(y)
mustart{T<:FloatingPoint}(d::Poisson,y::Vector{T},wt::Vector{T}) = map(Add(), y, convert(T,0.1))

xlogx(x::Real) = x == 0.0 ? 0.0 : x * log(x)
xlogxdmu(x::Real, mu::Real) = x == 0.0 ? 0.0 : x * log(x / mu)

type BernoulliDevResid <: TernaryFunctor end
function evaluate{T<:FloatingPoint}(::BernoulliDevResid,y::T,mu::T,wt::T)
    omy = one(T)-y
    2.wt*(xlogy(y,y/mu) + xlogy(omy,omy/(one(T)-mu)))
end
result_type{T<:FloatingPoint}(::BernoulliDevResid,::Type{T},::Type{T},::Type{T}) = T

type PoissonDevResid <: TernaryFunctor end
evaluate{T<:FloatingPoint}(::PoissonDevResid,y::T,mu::T,wt::T) = 2.wt * (xlogy(y,y/mu) - (y - mu))
result_type{T<:FloatingPoint}(::PoissonDevResid,::Type{T},::Type{T},::Type{T}) = T

function devresid!{T<:FloatingPoint}(d::Bernoulli,dr::Vector{T},y::Vector{T},
                                     mu::Vector{T},wt::Vector{T})
    map!(BernoulliDevResid(), dr, y, mu, wt)
end
function devresid!{T<:FloatingPoint}(d::Poisson,dr::Vector{T},y::Vector{T},
                                     mu::Vector{T},wt::Vector{T})
    map!(PoissonDevResid(), dr, y, mu, wt)
end

type BernoulliLogPdf <: BinaryFunctor end
function evaluate{T<:FloatingPoint}(::BernoulliLogPdf, y::T, mu::T)
    (y == zero(T) ? log(one(T) - mu) : (y == one(T) ? log(mu) : -inf(T)))
end
result_type{T<:FloatingPoint}(::BernoulliLogPdf,::Type{T},::Type{T}) = T

type PoissonLogPdf <: BinaryFunctor end
function evaluate{T<:FloatingPoint}(::PoissonLogPdf, y::T, mu::T)
    ccall((:dpois,:libRmath), Cdouble, (Cdouble,Cdouble,Cint), y, mu, 1)
end
result_type{T<:FloatingPoint}(::PoissonLogPdf,::Type{T},::Type{T}) = Float64

function deviance{T<:FloatingPoint}(d::Bernoulli, mu::Vector{T}, y::Vector{T}, wt::Vector{T})
    -2. * wsum(wt, BernoulliLogPdf(), y, mu)
end

function deviance{T<:FloatingPoint}(d::Poisson, mu::Vector{T}, y::Vector{T}, wt::Vector{T})
    -2. * wsum(wt, PoissonLogPdf(), y, mu)
end

