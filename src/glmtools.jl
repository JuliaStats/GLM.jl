"""
    Link

An abstract type whose subtypes determine methods for [`linkfun`](@ref), [`linkinv`](@ref),
and [`mueta`](@ref).
"""
@compat abstract type Link end

"""
    Link01

An abstract type of [`Link`](@ref) for which the range of [`linkinv`](@ref) is (0, 1).  As
such, the [`linkinv`](@ref) can be regarded as the cumulative distribution function (`cdf`)
of a continuous distribution with support (-∞, ∞). Most `Link01` subtypes are defined by a
distribution like this.

Typically such a link is used with a `Bernoulli` or `Binomial` distribution for the response
and the [`glmvar`](@ref) method returns `μ * (1 - μ)`.  To avoid returning a variance of
zero, a [`linkinvcomplement`] method is defined to return a tuple `(min(μ, 1 - μ), c)` where
`c` is a `Bool` indicating if the complement, `1 - μ`, is being returned.
"""
@compat abstract type Link01 <: Link end

"""
    CauchitLink

A [`Link01`](@ref) corresponding to the standard Cauchy distribution,
[`Distributions.Cauchy`](@ref).
"""
type CauchitLink  <: Link01 end

"""
    CloglogLink

A [`Link01`](@ref) corresponding to the extreme value (or log-Wiebull) distribution.  The
inverse link is called the complementary log-log transformation, `1 - exp(-exp(η))`.
"""
type CloglogLink  <: Link01 end

"""
    IdentityLink

The canonical [`Link`](@ref) for the `Normal` distribution, defined as `μ = η`.
"""
type IdentityLink <: Link   end

"""
    InverseLink

The canonical [`Link`](@ref) for the `Gamma` distribution, defined as `μ = inv(η)`.
"""
type InverseLink  <: Link   end

"""
    LogitLink

The canonical [`Link01`](@ref) for the `Bernoulli` and `Binomial` distributions,
The inverse link, [`linkinv`](@ref), is the c.d.f. of the standard logistic distribution,
(`Distributions.Logistic()`).
"""
type LogitLink    <: Link01 end

"""
    LogLink

The canonical [`Link`](@ref) for the [`Poisson`](@ref) distribution, defined as `η = log(μ)`.
"""
type LogLink      <: Link   end

"""
    ProbitLink

A [`Link01`](@ref) whose [`linkinv`](@ref) is the c.d.f. of the standard normal
distribution, (`Distributions.Normal()`).
"""
type ProbitLink   <: Link01 end

"""
    SqrtLink

A [`Link`](@ref) for which the [`linkfun`] is `η = √μ`
"""
type SqrtLink     <: Link   end

# x + 1 preserving the type of x
xp1(x) = x + one(x)

# default inverse link for cases where inverselinkorcomplement is defined
function linkinv(L::Link01, η)
    μ, c = linkinvcomplement(L, η)
    c ? one(μ) - μ : μ
end

linkfun(          ::CauchitLink, μ) = tan(pi * (μ - oftype(μ, 1/2)))
linkinvcomplement(::CauchitLink, η) = (oftype(η, 1/2) + atan(η) / pi, false)
mueta(            ::CauchitLink, η) = inv(pi * xp1(abs2(η)))

linkfun(          ::CloglogLink, μ) = log(-log1p(-μ))
linkinvcomplement(::CloglogLink, η) = η < 0 ? (-expm1(-exp(η)), false) : (exp(-exp(η)), true)
mueta(            ::CloglogLink, η) = exp(η) * exp(-exp(η))

linkfun(::IdentityLink, μ) = μ
linkinv(::IdentityLink, η) = η
mueta(  ::IdentityLink, η) = one(η)

linkfun(::InverseLink, μ) = inv(μ)
linkinv(::InverseLink, η) = inv(η)
mueta(  ::InverseLink, η) = -inv(abs2(η))

linkfun(          ::LogitLink, μ) = log(μ / (one(μ) - μ))
linkinvcomplement(::LogitLink, η) = (inv(xp1(exp(abs(η)))), η > 0)
function mueta(   ::LogitLink, η)
    e = exp(abs(η))
    e / abs2(xp1(e))
end

linkfun(::LogLink, μ) = log(μ)
linkinv(::LogLink, η) = exp(η)
mueta(  ::LogLink, η) = exp(η)

linkfun(          ::ProbitLink, μ) = -sqrt2 * erfcinv(2μ)
linkinvcomplement(::ProbitLink, η) = (erfc(abs(η) / sqrt2) / 2, η > 0)
mueta(            ::ProbitLink, η) = exp(-abs2(η) / 2) / sqrt2π

linkfun(::SqrtLink, μ) = sqrt(μ)
linkinv(::SqrtLink, η) = abs2(η)
mueta(  ::SqrtLink, η) = 2η

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
    linkinvcomplement(L::Link01, η)

Return a tuple of `μ`, the mean value, or its complement, `1 - μ`, for link `L` at linear
predictor value `η` plus a [`Bool`](@ref) indicator for the complement.
"""
function linkinv end

"""
    mueta(L::Link, η)

Return the derivative of [`linkinv`](@ref), `dμ/dη`, for link `L` at linear predictor value `η`.
"""
function mueta end

"""
    canonicallink(D::Distribution)

Return the canonical link for distribution `D`, which must be in the exponential family.
"""
function canonicallink end

canonicallink(::Bernoulli) = LogitLink()
canonicallink(::Binomial)  = LogitLink()
canonicallink(::Gamma)     = InverseLink()
canonicallink(::Normal)    = IdentityLink()
canonicallink(::Poisson)   = LogLink()

"""
    glmvar(D::Distribution, μ)

Return the variance function for distribution `D` evaluated at mean value `μ`.

The variance function is the part of the variance that depends upon `μ`.  The
variance of the response may be a multiple of this value but the multiple cannot
depend upon `μ`.
"""
glmvar(::Union{Bernoulli,Binomial}, μ) = μ * (1 - μ)
glmvar(::Gamma,                     μ) = abs2(μ)
glmvar(::Normal,                    μ) = one(μ)
glmvar(::Poisson,                   μ) = μ

"""
    mustart(D::Distribution, y, wt)

Return a suitable starting value for `μ` in distribution `D` given the response `y`.

Often a suitable value is `y`.  These methods exist to handle "corner cases" where, e.g.
the response `y` may be 0 but `μ` must be positive.

The `wt` argument is only used when `D` is `Binomial`.  For the purposes of GLMs a
`Binomial` response is the proportion of successes in `n` trials and `n` is used as a
case weight.
"""
mustart(::Bernoulli, y, wt) = (y + oftype(y, 1/2)) / 2
mustart(::Binomial , y, wt) = (wt * y + oftype(y, 1/2)) / xp1(wt)
mustart(::Gamma    , y, wt) = y == 0 ? oftype(y, 0.1) : y
mustart(::Normal   , y, wt) = y
mustart(::Poisson  , y, wt) = y + oftype(y, 0.1)

"""
    devresid(D::Distribution, y, μ)

Return the squared deviance residual of `y` and `μ` for distribution `D`.

The deviance for the parameters in a GLM is the sum of these squared deviance residuals
evaluated at the current `β`.

This name, matching that in R, is confusing because the deviance residuals are
`sign(y - μ) * sqrt(devresid(D, y, μ))`.
"""
function devresid(::Bernoulli, y, μ)
    if y == 1  # change to isone when lower bound on Julia version is 0.7
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

"""
    loglik_obs(D::Distribution, y, μ, wt, ϕ)

Return the log-likelihood contribution for `y` under distribution `D` with mean `μ`,
scale parameter `ϕ` and case weight `wt`.
"""
loglik_obs(::Bernoulli, y, μ, wt, ϕ) = wt*logpdf(Bernoulli(μ), y)
loglik_obs(::Binomial , y, μ, wt, ϕ) = logpdf(Binomial(Int(wt), μ), Int(y*wt))
loglik_obs(::Gamma    , y, μ, wt, ϕ) = wt*logpdf(Gamma(1/ϕ, μ*ϕ), y)
loglik_obs(::Normal   , y, μ, wt, ϕ) = wt*logpdf(Normal(μ, sqrt(ϕ)), y)
loglik_obs(::Poisson  , y, μ, wt, ϕ) = wt*logpdf(Poisson(μ), y)
