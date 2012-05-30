## Create link types.  For each type there should be a method for
## linkfun, linkinv and mueta.  The valideta and validmu functions
## should return the numeric argument or throw an error

chkpositive(x::Real) = isfinite(x) && 0. < x ? x : error("argument must be positive")
chkfinite(x::Real) = isfinite(x) ? x : error("argument must be finite")
chk01(x::Real) = 0. < x < 1. ? x : error("argument must be in (0, 1)")

abstract Link

type CauchitLink  <: Link end
linkfun (l::CauchitLink,   mu::Real) = tan(pi * (mu - 0.5))
linkinv (l::CauchitLink,  eta::Real) = 0.5 + atan(eta) / pi
mueta   (l::CauchitLink,  eta::Real) = 1. /(pi * (1 + eta * eta))
valideta(l::CauchitLink,  eta::Real) = chkfinite(eta)
validmu (l::CauchitLink,   mu::Real) = chk01(mu)

type CloglogLink  <: Link end
linkfun (l::CloglogLink,   mu::Real) = log(-log(1. - mu))
linkinv (l::CloglogLink,  eta::Real) = -expm1(-exp(eta))
mueta   (l::CloglogLink,  eta::Real) = exp(eta) * exp(-exp(eta))
const llmaxabs = log(-log(realmin(Float64)))
valideta(l::CloglogLink,  eta::Real) = abs(eta) < llmaxabs? eta: error("require abs(eta) < $llmaxab")
validmu (l::CauchitLink,   mu::Real) = chk01(mu)

type IdentityLink <: Link end
linkfun (l::IdentityLink,  mu::Real) = mu
linkinv (l::IdentityLink, eta::Real) = eta
mueta   (l::IdentityLink, eta::Real) = 1.
valideta(l::IdentityLink, eta::Real) = chkfinite(eta)
validmu (l::IdentityLink,  mu::Real) = chkfinite(mu)

type InverseLink  <: Link end
linkfun (l::InverseLink,   mu::Real) =  1. / mu
linkinv (l::InverseLink,  eta::Real) =  1. / eta
mueta   (l::InverseLink,  eta::Real) = -1. / (eta * eta)
valideta(l::InverseLink,  eta::Real) = chkpositive(eta)
validmu (l::InverseLink,  eta::Real) = chkpositive(mu)

type LogitLink    <: Link end
linkfun (l::LogitLink,     mu::Real) = log(mu / (1 - mu))
linkinv (l::LogitLink,    eta::Real) = 1. / (1. + exp(-eta))
mueta   (l::LogitLink,    eta::Real) = (e = exp(-abs(eta)); f = 1. + e; e / (f * f))
valideta(l::LogitLink,    eta::Real) = chkfinite(eta)
validmu (l::LogitLink,     mu::Real) = chk01(mu)

type LogLink      <: Link end
linkfun (l::LogLink,       mu::Real) = log(mu)
linkinv (l::LogLink,      eta::Real) = exp(eta)
mueta   (l::LogLink,      eta::Real) = exp(eta)
valideta(l::LogLink,      eta::Real) = chkfinite(eta)
validmu (l::LogLink,       mu::Real) = chkfinite(mu)

type ProbitLink   <: Link end
linkfun (l::ProbitLink,    mu::Real) =
    ccall(dlsym(_jl_libRmath, :qnorm5), Float64,
          (Float64,Float64,Float64,Int32,Int32), mu, 0., 1., 1, 0)
linkinv (l::ProbitLink,   eta::Real) = (1. + erf(eta/sqrt(2.))) / 2.
mueta   (l::ProbitLink,   eta::Real) = exp(-0.5eta^2) / sqrt(2.pi)
valideta(l::ProbitLink,   eta::Real) = chkfinite(eta)
validmu (l::ProbitLink,    mu::Real) = chk01(mu)

## Vectorized methods, including validity checks
function linkfun{T<:Real}(l::Link, mu::AbstractArray{T})
    eta = similar(mu, Float64)
    for i=1:numel(mu)
        eta[i] = linkfun(l, validmu(l, mu[i]))
    end
    eta
end
function linkinv{T<:Real}(l::Link, eta::AbstractArray{T})
    mu = similar(eta, Float64)
    for i=1:numel(eta)
        mu[i] = linkinv(l, valideta(l, eta[i]))
    end
    mu
end
function mueta{T<:Real}(l::Link, eta::AbstractArray{T})
    muEta = similar(eta, Float64)
    for i=1:numel(eta)
        muEta[i] = mueta(l, valideta(l, eta[i]))
    end
    muEta
end
