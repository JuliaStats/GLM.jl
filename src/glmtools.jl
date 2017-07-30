"""
    Link

An abstract type whose subtypes determine methods for [`linkfun`](@ref), [`linkinv`](@ref),
[`mueta`](@ref), and [`inverselink`](@ref).
"""
@compat abstract type Link end     # Link types define linkfun!, linkinv!, and mueta!

"""
    CauchitLink

A [`Link`](@ref) corresponding to the standard Cauchy distribution,
[`Distributions.Cauchy`](@ref).
"""
type CauchitLink <: Link end

"""
    CloglogLink

A [`Link`](@ref) corresponding to the extreme value (or log-Wiebull) distribution.  The
inverse link is called the complementary log-log transformation, `1 - exp(-exp(η))`.
"""
type CloglogLink  <: Link end

"""
    IdentityLink

The canonical [`Link`](@ref) for the `Normal` distribution, defined as `η = μ`.
"""
type IdentityLink <: Link end

"""
    InverseLink

The canonical [`Link`](@ref) for [`Distributions.Gamma`](@ref) distribution, defined as `η = inv(μ)`.
"""
type InverseLink  <: Link end

"""
    LogitLink

The canonical [`Link`](@ref) for [`Distributions.Bernoulli`](@ref) and [`Distributions.Binomial`](@ref).
The inverse link, [`linkinv`](@ref), is the c.d.f. of the standard logistic distribution,
[`Distributions.Logistic`](@ref).
"""
type LogitLink <: Link end

"""
    LogLink

The canonical [`Link`](@ref) for [`Distributions.Poisson`](@ref), defined as `η = log(μ)`.
"""
type LogLink <: Link end

"""
    ProbitLink

A [`Link`](@ref) whose [`linkinv`](@ref) is the c.d.f. of the standard normal
distribution, ()`Distributions.Normal()`).
"""
type ProbitLink <: Link end

"""
    SqrtLink

A [`Link`](@ref) defined as `η = √μ`
"""
type SqrtLink <: Link end

"""
    linkfun(L::Link, μ)

Return `η`, the value of the linear predictor for link `L` at mean `μ`.
"""
function linkfun end

"""
    linkinv(L::Link, η)

Return `μ`, the mean value, for link `L` at linear predictor value `η`.
"""
function linkinv end

"""
    mueta(L::Link, η)

Return the derivative of [`linkinv`](@ref), `dμ/dη`, for link `L` at linear predictor value `η`.
"""
function mueta end

"""
    inverselink(L::Link, η)

Return a 3-tuple of the inverse link, the derivative of the inverse link, and when appropriate, the variance function `μ*(1 - μ)`.

The variance function is returned as NaN unless the range of μ is (0, 1)
"""
function inverselink end

"""
    canonicallink(D::Distribution)

Return the canonical link for distribution `D`, which must be in the exponential family.
"""
function canonicallink end

linkfun(::CauchitLink, μ) = tan(pi * (μ - oftype(μ, 1/2)))
linkinv(::CauchitLink, η) = oftype(η, 1/2) + atan(η) / pi
mueta(::CauchitLink, η) = one(η) / (pi * (one(η) + abs2(η)))
function inverselink(::CauchitLink, η)
    μlower = atan(-abs(η)) / π
    μlower += oftype(μlower, 1/2)
    η > 0 ? 1 - μlower : μlower, π * (1 + abs2(η)), μlower * (1 + μlower)
end


linkfun(::CloglogLink, μ) = log(-log1p(-μ))
function linkinv{T<:Real}(::CloglogLink, η::T)
    clamp(-expm1(-exp(η)), eps(T), one(T) - eps(T))
end
function mueta{T<:Real}(::CloglogLink, η::T)
    max(eps(T), exp(η) * exp(-exp(η)))
end
function inverselink(::CloglogLink, η)
    expη = exp(η)
    μ = -expm1(-expη)
    omμ = exp(-expη)   # the complement, 1 - μ
    μ, expη * omμ, μ * omμ
end

linkfun(::IdentityLink, μ) = μ
linkinv(::IdentityLink, η) = η
mueta(::IdentityLink, η) = one(η)
linkinverse(::IdentityLink, η) = η, one(η), oftype(η, NaN)

linkfun(::InverseLink, μ) = inv(μ)
linkinv(::InverseLink, η) = inv(η)
mueta(::InverseLink, η) = -inv(abs2(η))
function linkinverse(::InverseLink, η)
    μ = inv(η)
    μ, -abs2(μ), oftype(μ, NaN)
end

linkfun(::LogitLink, μ) = logit(μ)
linkinv(::LogitLink, η) = logistic(η)
function mueta(::LogitLink, η)
    expabs = exp(-abs(η))
    denom = 1 + expabs
    (expabs / denom) / denom
end
function inverselink(::LogitLink, η)
    expabs = exp(-abs(η))
    opexpabs = 1 + expabs
    deriv = (expabs / opexpabs) / opexpabs
    η ≤ 0 ? expabs / opexpabs : inv(opexpabs), deriv, deriv
end

linkfun(::LogLink, μ) = log(μ)
linkinv(::LogLink, η) = exp(η)
mueta(::LogLink, η) = exp(η)
function inverselink(::LogLink, η)
    μ = exp(η)
    μ, μ, oftype(μ, NaN)
end

linkfun(::ProbitLink, μ) = -sqrt2 * erfcinv(2μ)
linkinv(::ProbitLink, η) = erfc(-η / sqrt2) / 2
mueta(::ProbitLink, η) = exp(-abs2(η) / 2) / sqrt2π
function inverselink(::ProbitLink)
    μlower = erfc(abs(η) / sqrt2) / 2
    μupper = 1 - μlower
    η < 0 ? μlower : μupper, exp(-abs2(η) / 2 ) / sqrt2π, μlower * μupper
end

linkfun(::SqrtLink, μ) = sqrt(μ)
linkinv(::SqrtLink, η) = abs2(η)
mueta(::SqrtLink, η) = 2η
inverselink(::SqrtLink, η) = abs2(η), 2η, oftype(η, NaN)

canonicallink(::Bernoulli) = LogitLink()
canonicallink(::Binomial) = LogitLink()
canonicallink(::Gamma) = InverseLink()
canonicallink(::Normal) = IdentityLink()
canonicallink(::Poisson) = LogLink()

# For the "odd" link functions we evaluate the linear predictor such that mu is closest to zero where the precision is higher
function glmvar(::Union{Bernoulli,Binomial}, link::Union{CauchitLink,InverseLink,LogitLink,ProbitLink}, μ, η)
    μ = linkinv(link, ifelse(η < 0, η, -η))
    μ * (1 - μ)
end
glmvar(::Union{Bernoulli,Binomial}, ::Link, μ, η) = μ * (1 - μ)
glmvar(::Gamma, ::Link, μ, η) = abs2(μ)
glmvar(::Normal, ::Link, μ, η) = 1
glmvar(::Poisson, ::Link, μ, η) = μ

mustart(::Bernoulli, y, wt) = (y + oftype(y, 0.5)) / 2
mustart(::Binomial, y, wt) = (wt * y + oftype(y, 0.5)) / (wt + one(y))
mustart(::Gamma, y, wt) = y == 0 ? oftype(y, 0.1) : y
mustart(::Normal, y, wt) = y
mustart(::Poisson, y, wt) = y + oftype(y, 0.1)

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
devresid(::Gamma, y, μ) = -2 * (log(y / μ) - (y - μ) / μ)
devresid(::Normal, y, μ) = abs2(y - μ)
devresid(::Poisson, y, μ) = 2 * (xlogy(y, y / μ) - (y - μ))

# Whether a dispersion parameter has to be estimated for a distribution
dispersion_parameter(::Union{Bernoulli, Binomial, Poisson}) = false
dispersion_parameter(::Union{Gamma, Normal, InverseGaussian}) = true

# Log-likelihood for an observation
loglik_obs(::Bernoulli, y, μ, wt, ϕ) = wt*logpdf(Bernoulli(μ), y)
loglik_obs(::Binomial, y, μ, wt, ϕ) = logpdf(Binomial(Int(wt), μ), Int(y*wt))
loglik_obs(::Gamma, y, μ, wt, ϕ) = wt*logpdf(Gamma(1/ϕ, μ*ϕ), y)
loglik_obs(::Normal, y, μ, wt, ϕ) = wt*logpdf(Normal(μ, sqrt(ϕ)), y)
loglik_obs(::Poisson, y, μ, wt, ϕ) = wt*logpdf(Poisson(μ), y)
