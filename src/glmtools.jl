@compat abstract type Link end     # Link types define linkfun!, linkinv!, and mueta!

type CauchitLink  <: Link end
type CloglogLink  <: Link end
type IdentityLink <: Link end
type InverseLink  <: Link end
type LogitLink    <: Link end
type LogLink      <: Link end
type ProbitLink   <: Link end
type SqrtLink     <: Link end

linkfun(::CauchitLink, μ) = tan(pi * (μ - oftype(μ, 1/2)))
linkinv(::CauchitLink, η) = (oftype(η, 1/2) + atan(η) / pi, false)
mueta(  ::CauchitLink, η) = one(η) / (pi * (one(η) + abs2(η)))

linkfun(::CloglogLink, μ) = log(-log1p(-μ))
linkinv(::CloglogLink, η) = η < 0 ? (-expm1(-exp(η)), false) : (exp(-exp(η)), true)
mueta(  ::CloglogLink, η) = exp(η) * exp(-exp(η))

linkfun(::IdentityLink, μ) = μ
linkinv(::IdentityLink, η) = (η, false)
mueta(  ::IdentityLink, η) = 1

linkfun(::InverseLink, μ) = inv(μ)
linkinv(::InverseLink, η) = (inv(η), false)
mueta(  ::InverseLink, η) = -inv(abs2(η))

linkfun(::LogitLink, μ) = logit(μ)
linkinv(::LogitLink, η) = (logistic(-abs(η)), η > 0)
function mueta(::LogitLink, η)
    e = exp(-abs(η))
    f = one(η) + e
    return e / (f * f)
end

linkfun(::LogLink, μ) = log(μ)
linkinv(::LogLink, η) = (exp(η), false)
mueta(  ::LogLink, η) = exp(η)

linkfun(::ProbitLink, μ) = -sqrt2 * erfcinv(2μ)
linkinv(::ProbitLink, η) = (erfc(abs(η) / sqrt2) / 2, η > 0)
mueta(  ::ProbitLink, η) = exp(-abs2(η) / 2) / sqrt2π

linkfun(::SqrtLink, μ) = sqrt(μ)
linkinv(::SqrtLink, η) = (abs2(η), false)
mueta(  ::SqrtLink, η) = 2η

canonicallink(::Bernoulli) = LogitLink()
canonicallink(::Binomial)  = LogitLink()
canonicallink(::Gamma)     = InverseLink()
canonicallink(::Normal)    = IdentityLink()
canonicallink(::Poisson)   = LogLink()

glmvar(::Union{Bernoulli,Binomial}, ::Link, μ) = μ * (1 - μ)
glmvar(::Gamma,                     ::Link, μ) = abs2(μ)
glmvar(::Normal,                    ::Link, μ) = 1
glmvar(::Poisson,                   ::Link, μ) = μ

mustart(::Bernoulli, y, wt) = (y + oftype(y, 0.5)) / 2
mustart(::Binomial , y, wt) = (wt * y + oftype(y, 0.5)) / (wt + one(y))
mustart(::Gamma    , y, wt) = y == 0 ? oftype(y, 0.1) : y
mustart(::Normal   , y, wt) = y
mustart(::Poisson  , y, wt) = y + oftype(y, 0.1)

function devresid(::Bernoulli, y, μ)
    if y == 1
        return -2 * log(μ)
    elseif y == 0
        return -2 * log1p(-μ)
    end
    throw(ArgumentError("y should be 0 or 1 (got $y)"))
end
function devresid(::Binomial, y, μ)
    if y == 1
        return -2 * log(μ)
    elseif y == 0
        return -2 * log1p(-μ)
    else
        return 2 * (y * (log(y) - log(μ)) + (1 - y)*(log1p(-y) - log1p(-μ)))
    end
end
devresid(::Gamma  , y, μ) = -2 * (log(y / μ) - (y - μ) / μ)
devresid(::Normal , y, μ) = abs2(y - μ)
devresid(::Poisson, y, μ) = 2 * (xlogy(y, y / μ) - (y - μ))

# Whether a dispersion parameter has to be estimated for a distribution
dispersion_parameter(::Union{Bernoulli, Binomial, Poisson})   = false
dispersion_parameter(::Union{Gamma, Normal, InverseGaussian}) = true

# Log-likelihood for an observation
loglik_obs(::Bernoulli, y, μ, wt, ϕ) = wt*logpdf(Bernoulli(μ), y)
loglik_obs(::Binomial , y, μ, wt, ϕ) = logpdf(Binomial(Int(wt), μ), Int(y*wt))
loglik_obs(::Gamma    , y, μ, wt, ϕ) = wt*logpdf(Gamma(1/ϕ, μ*ϕ), y)
loglik_obs(::Normal   , y, μ, wt, ϕ) = wt*logpdf(Normal(μ, sqrt(ϕ)), y)
loglik_obs(::Poisson  , y, μ, wt, ϕ) = wt*logpdf(Poisson(μ), y)
