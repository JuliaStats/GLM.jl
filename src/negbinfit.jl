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
    negbin(formula, data, [link::Link];
           <keyword arguments>)
    negbin(X::AbstractMatrix, y::AbstractVector, [link::Link];
           <keyword arguments>)

Fit a negative binomial generalized linear model to data, while simultaneously
estimating the shape parameter θ. Extra arguments and keyword arguments will be
passed to [`glm`](@ref).

In the first method, `formula` must be a
[StatsModels.jl `Formula` object](https://juliastats.org/StatsModels.jl/stable/formula/)
and `data` a table (in the [Tables.jl](https://tables.juliadata.org/stable/) definition, e.g. a data frame).
In the second method, `X` must be a matrix holding values of the independent variable(s)
in columns (including if appropriate the intercept), and `y` must be a vector holding
values of the dependent variable.
In both cases, `link` may specify the link function
(if omitted, it is taken to be `NegativeBinomial(θ)`).

# Keyword Arguments
- `initialθ::Real=Inf`: Starting value for shape parameter θ. If it is `Inf`
  then the initial value will be estimated by fitting a Poisson distribution.
- `dropcollinear::Bool=true`: See `dropcollinear` for [`glm`](@ref)
- `method::Symbol=:qr`: See `method` for [`glm`](@ref)
- `maxiter::Integer=30`: See `maxiter` for [`glm`](@ref)
- `atol::Real=1.0e-6`: See `atol` for [`glm`](@ref)
- `rtol::Real=1.0e-6`: See `rtol` for [`glm`](@ref)
"""
function negbin(F,
                D,
                args...;
                wts::Union{AbstractWeights,AbstractVector{<:Real}}=uweights(0),
                initialθ::Real=Inf,
                dropcollinear::Bool=true,
                method::Symbol=:qr,
                maxiter::Integer=30,
                minstepfac::Real=0.001,
                atol::Real=1e-6,
                rtol::Real=1.e-6,
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

    # fit a Poisson regression model if the user does not specify an initial θ
    if isinf(initialθ)
        regmodel = glm(F, D, Poisson(), args...;
                       wts=wts, dropcollinear=dropcollinear, method=method, maxiter=maxiter,
                       atol=atol, rtol=rtol, kwargs...)
    else
        regmodel = glm(F, D, NegativeBinomial(initialθ), args...;
                       wts=wts, dropcollinear=dropcollinear, method=method, maxiter=maxiter,
                       atol=atol, rtol=rtol, kwargs...)
    end

    μ = regmodel.rr.mu
    y = regmodel.rr.y
    wts = regmodel.rr.wts
    lw, ly = length(wts), length(y)
    if lw != ly
        throw(ArgumentError("length of `wts` must be $ly but was $lw"))
    end

    θ = mle_for_θ(y, μ, wts; maxiter=maxiter, tol=rtol)
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
        regmodel = glm(F, D, NegativeBinomial(θ), args...;
                       dropcollinear=dropcollinear, method=method, maxiter=maxiter,
                       atol=atol, rtol=rtol, kwargs...)
        μ = regmodel.rr.mu
        prevθ = θ
        θ = mle_for_θ(y, μ, wts; maxiter=maxiter, tol=rtol)
        δ = prevθ - θ
        ll0 = ll
        ll = loglikelihood(regmodel)
    end
    converged || throw(ConvergenceException(maxiter))
    return regmodel
end
