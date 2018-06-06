"""
    Link

An abstract type whose subtypes determine methods for [`linkfun`](@ref), [`linkinv`](@ref),
[`mueta`](@ref), and [`inverselink`](@ref).
"""
abstract type Link end

"""
    Link01

An abstract subtype of [`Link`](@ref) which are links defined on (0, 1)
"""
abstract type Link01 <: Link end

"""
    CauchitLink

A [`Link01`](@ref) corresponding to the standard Cauchy distribution,
[`Distributions.Cauchy`](https://juliastats.github.io/Distributions.jl/stable/univariate.html#Distributions.Cauchy).
"""
mutable struct CauchitLink <: Link01 end

"""
    CloglogLink

A [`Link01`](@ref) corresponding to the extreme value (or log-Wiebull) distribution.  The
link is the complementary log-log transformation, `log(1 - log(-μ))`.
"""
mutable struct CloglogLink  <: Link01 end

"""
    IdentityLink

The canonical [`Link`](@ref) for the `Normal` distribution, defined as `η = μ`.
"""
mutable struct IdentityLink <: Link end

"""
    InverseLink

The canonical [`Link`](@ref) for [`Distributions.Gamma`](https://juliastats.github.io/Distributions.jl/stable/univariate.html#Distributions.Gamma) distribution, defined as `η = inv(μ)`.
"""
mutable struct InverseLink  <: Link end

"""
    InverseSquareLink

The canonical [`Link`](@ref) for [`Distributions.InverseGaussian`](https://juliastats.github.io/Distributions.jl/stable/univariate.html#Distributions.InverseGaussian) distribution, defined as `η = inv(abs2(μ))`.
"""
mutable struct InverseSquareLink  <: Link end

"""
    LogitLink

The canonical [`Link01`](@ref) for [`Distributions.Bernoulli`](https://juliastats.github.io/Distributions.jl/stable/univariate.html#Distributions.Bernoulli) and [`Distributions.Binomial`](https://juliastats.github.io/Distributions.jl/stable/univariate.html#Distributions.Binomial).
The inverse link, [`linkinv`](@ref), is the c.d.f. of the standard logistic distribution,
[`Distributions.Logistic`](https://juliastats.github.io/Distributions.jl/stable/univariate.html#Distributions.Logistic).
"""
mutable struct LogitLink <: Link01 end

"""
    LogLink

The canonical [`Link`](@ref) for [`Distributions.Poisson`](https://juliastats.github.io/Distributions.jl/stable/univariate.html#Distributions.Poisson), defined as `η = log(μ)`.
"""
mutable struct LogLink <: Link end

"""
    NegativeBinomialLink

The canonical [`Link`](@ref) for [`Distributions.NegativeBinomial`](@ref) distribution, defined as `η = log(μ/(μ+θ))`.
θ has to be fixed for the distribution to belong to the exponential family
"""
mutable struct NegativeBinomialLink  <: Link
    θ::Float64
end

"""
    ProbitLink

A [`Link01`](@ref) whose [`linkinv`](@ref) is the c.d.f. of the standard normal
distribution, [`Distributions.Normal()`](https://juliastats.github.io/Distributions.jl/stable/univariate.html#Distributions.Normal).
"""
mutable struct ProbitLink <: Link01 end

"""
    SqrtLink

A [`Link`](@ref) defined as `η = √μ`
"""
mutable struct SqrtLink <: Link end

"""
    linkfun(L::Link, μ)

Return `η`, the value of the linear predictor for link `L` at mean `μ`.

# Examples
```jldoctest
julia> μ = inv(10):inv(5):1
0.1:0.2:0.9

julia> show(linkfun.(LogitLink(), μ))
[-2.19722, -0.847298, 0.0, 0.847298, 2.19722]

```
"""
function linkfun end

"""
    linkinv(L::Link, η)

Return `μ`, the mean value, for link `L` at linear predictor value `η`.

# Examples
```jldoctest
julia> μ = inv(10):inv(5):1; showcompact(collect(μ))
[0.1, 0.3, 0.5, 0.7, 0.9]
julia> η = logit.(μ); showcompact(η)
[-2.19722, -0.847298, 0.0, 0.847298, 2.19722]
julia> showcompact(linkinv.(LogitLink(), η))
[0.1, 0.3, 0.5, 0.7, 0.9]
```
"""
function linkinv end

"""
    mueta(L::Link, η)

Return the derivative of [`linkinv`](@ref), `dμ/dη`, for link `L` at linear predictor value `η`.

# Examples
```jldoctest
julia> showcompact(mueta(LogitLink(), 0.0))
0.25
julia> showcompact(mueta(CloglogLink(), 0.0))
0.367879
julia> showcompact(mueta(LogLink(), 2.0))
7.38906
```
"""
function mueta end

"""
    inverselink(L::Link, η)

Return a 3-tuple of the inverse link, the derivative of the inverse link, and when appropriate, the variance function `μ*(1 - μ)`.

The variance function is returned as NaN unless the range of μ is (0, 1)

# Examples
```jldoctest
julia> showcompact(inverselink(LogitLink(), 0.0))
(0.5, 0.25, 0.25)
julia> showcompact(inverselink(CloglogLink(), 0.0))
(0.632121, 0.367879, 0.232544)
julia> showcompact(inverselink(LogLink(), 2.0))
(7.38906, 7.38906, NaN)
```
"""
function inverselink end

"""
    canonicallink(D::Distribution)

Return the canonical link for distribution `D`, which must be in the exponential family.

# Examples
```jldoctest
julia> canonicallink(Bernoulli())
GLM.LogitLink()
```
"""
function canonicallink end

linkfun(::CauchitLink, μ) = tan(pi * (μ - oftype(μ, 1/2)))
linkinv(::CauchitLink, η) = oftype(η, 1/2) + atan(η) / pi
mueta(::CauchitLink, η) = one(η) / (pi * (one(η) + abs2(η)))
function inverselink(::CauchitLink, η)
    μlower = atan(-abs(η)) / π
    μlower += oftype(μlower, 1/2)
    η > 0 ? 1 - μlower : μlower, inv(π * (1 + abs2(η))), μlower * (1 - μlower)
end

linkfun(::CloglogLink, μ) = log(-log1p(-μ))
function linkinv(::CloglogLink, η::T) where T<:Real
    clamp(-expm1(-exp(η)), eps(T), one(T) - eps(T))
end
function mueta(::CloglogLink, η::T) where T<:Real
    max(eps(T), exp(η) * exp(-exp(η)))
end
function inverselink(::CloglogLink, η)
    expη = exp(η)
    μ = -expm1(-expη)
    omμ = exp(-expη)   # the complement, 1 - μ
    μ, max(realmin(μ), expη * omμ), max(realmin(μ), μ * omμ)
end

linkfun(::IdentityLink, μ) = μ
linkinv(::IdentityLink, η) = η
mueta(::IdentityLink, η) = one(η)
inverselink(::IdentityLink, η) = η, one(η), oftype(η, NaN)

linkfun(::InverseLink, μ) = inv(μ)
linkinv(::InverseLink, η) = inv(η)
mueta(::InverseLink, η) = -inv(abs2(η))
function inverselink(::InverseLink, η)
    μ = inv(η)
    μ, -abs2(μ), oftype(μ, NaN)
end

linkfun(::InverseSquareLink, μ) = inv(abs2(μ))
linkinv(::InverseSquareLink, η) = inv(sqrt(η))
mueta(::InverseSquareLink, η) = -inv(2η*sqrt(η))
function inverselink(::InverseSquareLink, η)
    μ = inv(sqrt(η))
    μ, -μ / (2η), oftype(μ, NaN)
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

linkfun(l::NegativeBinomialLink, μ) = log(μ/(μ+l.θ))
linkinv(l::NegativeBinomialLink, η) = e^η * l.θ / (1.0-e^η)
mueta(l::NegativeBinomialLink, η) = e^η * l.θ / (1.0-e^η)
function inverselink(l::NegativeBinomialLink, η)
    θ = l.θ
    μ = e^η * θ / (1-e^η)
    deriv = μ * (1.0 + μ/θ)
    μ, deriv, oftype(μ, NaN)
end

linkfun(::ProbitLink, μ) = -sqrt2 * erfcinv(2μ)
linkinv(::ProbitLink, η) = erfc(-η / sqrt2) / 2
mueta(::ProbitLink, η) = exp(-abs2(η) / 2) / sqrt2π
function inverselink(::ProbitLink, η)
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
canonicallink(::InverseGaussian) = InverseSquareLink()
canonicallink(::Normal) = IdentityLink()must
canonicallink(::Poisson) = LogLink()
canonicallink(d::NegativeBinomial) = NegativeBinomialLink(d.r)

"""
    glmvar(D::Distribution, μ)

Return the value of the variance function for `D` at `μ`

The variance of `D` at `μ` is the product of the dispersion parameter, ϕ, which does not
depend on `μ` and the value of `glmvar`.  In other words `glmvar` returns the factor of the
variance that depends on `μ`.

# Examples
```jldoctest
julia> μ = inv(6):inv(3):1; showcompact(collect(μ))
[0.166667, 0.5, 0.833333]
julia> showcompact(glmvar.(Normal(), μ))    # constant for Normal()
[1.0, 1.0, 1.0]
julia> showcompact(glmvar.(Bernoulli(), μ)) # μ * (1 - μ) for Bernoulli()
[0.138889, 0.25, 0.138889]
julia> showcompact(glmvar.(Poisson(), μ))   # μ for Poisson()
[0.166667, 0.5, 0.833333]
```
"""
function glmvar end

glmvar(::Union{Bernoulli,Binomial}, μ) = μ * (1 - μ)
glmvar(::Gamma, μ) = abs2(μ)
glmvar(::InverseGaussian, μ) = μ^3
glmvar(d::NegativeBinomial, μ) = μ * (1.0 + μ/d.r)
glmvar(::Normal, μ) = one(μ)
glmvar(::Poisson, μ) = μ

"""
    mustart(D::Distribution, y, wt)

Return a starting value for μ.

For some distributions it is appropriate to set `μ = y` to initialize the IRLS algorithm but
for others, notably the Bernoulli, the values of `y` are not allowed as values of `μ` and
must be modified.

# Examples
```jldoctest
julia> showcompact(mustart(Bernoulli(), 0.0, 1))
0.25
julia> showcompact(mustart(Bernoulli(), 1.0, 1))
0.75
julia> showcompact(mustart(Binomial(), 0.0, 10))
0.0454545
julia> showcompact(mustart(Normal(), 0.0, 1))
0.0
```
"""
function mustart end

mustart(::Bernoulli, y, wt) = (y + oftype(y, 1/2)) / 2
mustart(::Binomial, y, wt) = (wt * y + oftype(y, 1/2)) / (wt + one(y))
mustart(::Union{Gamma, InverseGaussian}, y, wt) = y == 0 ? oftype(y, 1/10) : y
mustart(::NegativeBinomial, y, wt) = y == 0 ? y + oftype(y, 1/6) : y
mustart(::Normal, y, wt) = y
mustart(::Poisson, y, wt) = y + oftype(y, 1/10)

"""
    devresid(D, y, μ)

Return the squared deviance residual of `μ` from `y` for distribution `D`

The deviance of a GLM can be evaluated as the sum of the squared deviance residuals.  This
is the principal use for these values.  The actual deviance residual, say for plotting, is
the signed square root of this value
```julia
sign(y - μ) * sqrt(devresid(D, y, μ))
```

# Examples
```jldoctest
julia> showcompact(devresid(Normal(), 0, 0.25))     # abs2(y - μ)
0.0625
julia> showcompact(devresid(Bernoulli(), 1, 0.75))  # -2log(μ) when y == 1
0.575364
julia> showcompact(devresid(Bernoulli(), 0, 0.25))  # -2log1p(-μ) = -2log(1-μ) when y == 0
0.575364
```
"""
function devresid end

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
devresid(::InverseGaussian, y, μ) = abs2(y - μ) / (y * abs2(μ))
function devresid(d::NegativeBinomial, y, μ)
    θ = d.r
    if μ == 0.0
        throw(ArgumentError("μ = 0.0 when computing deviance for NegativeBinomial"))
    end
    return 2 * (xlogy(y, y / μ) + xlogy(y + θ, (μ + θ)/(y + θ)))
end
devresid(::Normal, y, μ) = abs2(y - μ)
devresid(::Poisson, y, μ) = 2 * (xlogy(y, y / μ) - (y - μ))

"""
    dispersion_parameter(D)  # not exported

Does distribution `D` have a separate dispersion parameter, ϕ?

Returns `false` for the `Bernoulli`, `Binomial` and `Poisson` distributions, `true` otherwise.

# Examples
```jldoctest
julia> show(GLM.dispersion_parameter(Normal()))
true
julia> show(GLM.dispersion_parameter(Bernoulli()))
false
```
"""
dispersion_parameter(D) = true
dispersion_parameter(::Union{Bernoulli, Binomial, Poisson}) = false

"""
    loglik_obs(D, y, μ, wt, ϕ)  # not exported

Returns `wt * logpdf(D(μ, ϕ), y)` where the parameters of `D` are derived from `μ` and `ϕ`.

The `wt` argument is a multiplier of the result except in the case of the `Binomial` where
`wt` is the number of trials and `μ` is the proportion of successes.

The loglikelihood of a fitted model is the sum of these values over all the observations.
"""
function loglik_obs end

loglik_obs(::Bernoulli, y, μ, wt, ϕ) = wt*logpdf(Bernoulli(μ), y)
loglik_obs(::Binomial, y, μ, wt, ϕ) = logpdf(Binomial(Int(wt), μ), Int(y*wt))
loglik_obs(::Gamma, y, μ, wt, ϕ) = wt*logpdf(Gamma(inv(ϕ), μ*ϕ), y)
loglik_obs(::InverseGaussian, y, μ, wt, ϕ) = wt*logpdf(InverseGaussian(μ, inv(ϕ)), y)
loglik_obs(::Normal, y, μ, wt, ϕ) = wt*logpdf(Normal(μ, sqrt(ϕ)), y)
loglik_obs(::Poisson, y, μ, wt, ϕ) = wt*logpdf(Poisson(μ), y)
# μ = θp/(1-p) ==> p = μ/(μ+θ)
loglik_obs(d::NegativeBinomial, y, μ, wt, ϕ) = wt*logpdf(NegativeBinomial(d.r, μ/(μ+d.r)), y)
