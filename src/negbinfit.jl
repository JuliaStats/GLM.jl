function mle_for_θ(y::AbstractVector, μ::AbstractVector, wts::AbstractWeights;
                   maxiter=30, tol=1.e-6)
    function first_derivative(θ::Real)
        function tmp(yi, μi)
            return (yi + θ) / (μi + θ) + log(μi + θ) - 1 - log(θ) - digamma(θ + yi) +
                   digamma(θ)
        end
        return wts isa UnitWeights ? sum(tmp(yi, μi) for (yi, μi) in zip(y, μ)) :
               sum(wti * tmp(yi, μi) for (wti, yi, μi) in zip(wts, y, μ))
    end
    function second_derivative(θ::Real)
        function tmp(yi, μi)
            return -(yi + θ) / (μi + θ)^2 + 2 / (μi + θ) - 1 / θ - trigamma(θ + yi) +
                   trigamma(θ)
        end
        return wts isa UnitWeights ? sum(tmp(yi, μi) for (yi, μi) in zip(y, μ)) :
               sum(wti * tmp(yi, μi) for (wti, yi, μi) in zip(wts, y, μ))
    end

    if wts isa UnitWeights
        n = length(y)
        θ = n / sum((yi / μi - 1)^2 for (yi, μi) in zip(y, μ))
    else
        n = sum(wts)
        θ = n / sum(wti * (yi / μi - 1)^2 for (wti, yi, μi) in zip(wts, y, μ))
    end
    δ, converged = one(θ), false

    for t in 1:maxiter
        θ = abs(θ)
        δ = first_derivative(θ) / second_derivative(θ)
        if abs(δ) <= tol
            converged = true
            break
        end
        θ = θ - δ
    end
    if !converged
        info_msg = "Estimating dispersion parameter failed, which may " *
                   "indicate Poisson distributed data."
        throw(ConvergenceException(maxiter, NaN, NaN, info_msg))
    end
    return θ
end

"""
    negbin(formula::FormulaTerm, data, link::Union{Link,Nothing}=nothing;
           <keyword arguments>)
    negbin(X::AbstractMatrix, y::AbstractVector, link::Union{Link,Nothing}=nothing;
           <keyword arguments>)

Fit a negative binomial generalized linear model to data, while simultaneously
estimating the shape parameter θ. Arguments are the same as for [`glm`](@ref) except `initialθ`.

In the first method, `formula` must be a
[StatsModels.jl `Formula` object](https://juliastats.org/StatsModels.jl/stable/formula/)
and `data` a table (in the [Tables.jl](https://tables.juliadata.org/stable/) definition, e.g. a data frame).
In the second method, `X` must be a matrix holding values of the independent variable(s)
in columns (including if appropriate the intercept), and `y` must be a vector holding
values of the dependent variable.

In both cases, `link` may specify the link function. If omitted, it is taken to be
`NegativeBinomialLink(θ)`.

# Keyword Arguments
- `initialθ::Real=Inf`: Starting value for shape parameter θ. If it is `Inf`
  then the initial value will be estimated by fitting a Poisson distribution.
$COMMON_FIT_KWARGS_DOCS
- `offset::Union{AbstractVector{<:Real},Nothing}=nothing,`: offset added to `Xβ`
  to form `eta`.  Can be of length 0.
- `maxiter::Integer=30`: Maximum number of iterations allowed to achieve convergence
- `atol::Real=1e-6`: Convergence is achieved when the relative change in
  deviance is less than `max(rtol*dev, atol)`.
- `rtol::Real=1e-6`: Convergence is achieved when the relative change in
  deviance is less than `max(rtol*dev, atol)`.
- `minstepfac::Real=0.001`: Minimum line step fraction. Must be between 0 and 1.
- `start::Union{AbstractVector,Nothing}=nothing`: Starting values for beta. Should have the
  same length as the number of columns in the model matrix.
"""
function negbin(X::AbstractMatrix, y::AbstractVector, l::Union{Link,Nothing}=nothing;
                initialθ::Real=Inf,
                offset::Union{AbstractVector{<:Real},Nothing}=nothing,
                weights::AbstractVector{<:Real}=uweights(length(y)),
                dropcollinear::Bool=true,
                method::Symbol=:qr,
                maxiter::Integer=30,
                atol::Real=1e-6,
                rtol::Real=1e-6,
                minstepfac::Real=0.001,
                start::Union{AbstractVector,Nothing}=nothing,
                kwargs...)
    return _negbin(X, y, l;
                   initialθ, offset, weights, dropcollinear, method, contrasts=nothing,
                   maxiter, atol, rtol, minstepfac, start, kwargs...)
end
function negbin(formula::FormulaTerm, data, l::Union{Link,Nothing}=nothing;
                initialθ::Real=Inf,
                offset::Union{AbstractVector{<:Real},Nothing}=nothing,
                weights::Union{AbstractVector{<:Real},Symbol,AbstractString}=uweights(0),
                dropcollinear::Bool=true,
                method::Symbol=:qr,
                contrasts::AbstractDict{Symbol}=Dict{Symbol,Any}(),
                maxiter::Integer=30,
                atol::Real=1e-6,
                rtol::Real=1e-6,
                minstepfac::Real=0.001,
                start::Union{AbstractVector,Nothing}=nothing,
                kwargs...)
    return _negbin(formula, data, l;
                   initialθ, offset, weights, dropcollinear, method, contrasts,
                   maxiter, atol, rtol, minstepfac, start, kwargs...)
end

function _negbin(F,
                 D,
                 l::Union{Link,Nothing};
                 initialθ::Real,
                 offset::Union{AbstractVector{<:Real},Nothing},
                 weights::Union{AbstractVector{<:Real},Symbol,AbstractString},
                 dropcollinear::Bool,
                 method::Symbol,
                 contrasts::Union{AbstractDict{Symbol},Nothing},
                 maxiter::Integer,
                 atol::Real,
                 rtol::Real,
                 minstepfac::Real,
                 start::Union{AbstractVector,Nothing},
                 kwargs...)
    if haskey(kwargs, :verbose)
        Base.depwarn("""`verbose` argument is deprecated, use `ENV["JULIA_DEBUG"]=GLM` instead.""",
                     :negbin)
    end
    if !issubset(keys(kwargs), (:verbose,))
        throw(ArgumentError("unsupported keyword argument"))
    end

    maxiter >= 1 || throw(ArgumentError("maxiter must be positive"))
    atol > 0 || throw(ArgumentError("atol must be positive"))
    rtol > 0 || throw(ArgumentError("rtol must be positive"))
    initialθ > 0 || throw(ArgumentError("initialθ must be positive"))

    contrasts_kwarg = isnothing(contrasts) ? () : (contrasts=contrasts,)

    # fit a Poisson regression model if the user does not specify an initial θ
    distr = isinf(initialθ) ? Poisson() : NegativeBinomial(initialθ)
    regmodel = glm(F, D, distr, something(l, canonicallink(distr));
                   offset, weights, dropcollinear, method,
                   maxiter, atol, rtol, minstepfac, start, contrasts_kwarg...)

    μ = regmodel.rr.mu
    y = regmodel.rr.y
    weightsvec = regmodel.rr.weights

    θ = mle_for_θ(y, μ, weightsvec; maxiter=maxiter, tol=rtol)
    d = sqrt(2 * max(1, deviance(regmodel)))
    δ = one(θ)
    ll = loglikelihood(regmodel)
    ll0 = ll + 2 * d

    converged = false
    for i in 1:maxiter
        if abs(ll0 - ll) / d + abs(δ) <= rtol
            converged = true
            break
        end
        @debug "NegativeBinomial dispersion optimization" iteration = i θ = θ
        regmodel = glm(F, D, NegativeBinomial(θ), something(l, NegativeBinomialLink(θ));
                       offset, weights, dropcollinear, method,
                       maxiter, atol, rtol, minstepfac, start, contrasts_kwarg...)
        μ = regmodel.rr.mu
        prevθ = θ
        θ = mle_for_θ(y, μ, weightsvec; maxiter=maxiter, tol=rtol)
        δ = prevθ - θ
        ll0 = ll
        ll = loglikelihood(regmodel)
    end
    converged || throw(ConvergenceException(maxiter))
    return regmodel
end
