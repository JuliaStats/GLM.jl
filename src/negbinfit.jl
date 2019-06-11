function mle_for_θ(y::AbstractVector, μ::AbstractVector, wts::AbstractVector;
                   maxiter=30, tol=1.e-6)
    function first_derivative(θ::Real)
        tmp(yi, μi) = (yi+θ)/(μi+θ) + log(μi+θ) - 1 - log(θ) - digamma(θ+yi) + digamma(θ)
        unit_weights ? sum(tmp(yi, μi) for (yi, μi) in zip(y, μ)) :
                       sum(wti * tmp(yi, μi) for (wti, yi, μi) in zip(wts, y, μ))
    end
    function second_derivative(θ::Real)
        tmp(yi, μi) = -(yi+θ)/(μi+θ)^2 + 2/(μi+θ) - 1/θ - trigamma(θ+yi) + trigamma(θ)
        unit_weights ? sum(tmp(yi, μi) for (yi, μi) in zip(y, μ)) :
                       sum(wti * tmp(yi, μi) for (wti, yi, μi) in zip(wts, y, μ))
    end

    unit_weights = length(wts) == 0
    if unit_weights
        n = length(y) 
        θ = n / sum((yi/μi - 1)^2 for (yi, μi) in zip(y, μ))
    else
        n = sum(wts)
        θ = n / sum(wti * (yi/μi - 1)^2 for (wti, yi, μi) in zip(wts, y, μ))
    end
    δ, converged = one(θ), false

    for t = 1:maxiter
        θ = abs(θ)
        δ = first_derivative(θ) / second_derivative(θ)
        if abs(δ) <= tol
            converged = true
            break
        end
        θ = θ - δ
    end
    converged || throw(ConvergenceException(maxiter))
    θ
end

function negbin(F, D, args...;
                initialθ::Real=Inf, maxiter::Integer=30, tol::Real=1.e-6,
                verbose::Bool=false, kwargs...)
    if haskey(kwargs, :maxIter)
        Base.depwarn("'maxIter' argument is deprecated, use 'maxiter' instead", :fit!)
        maxiter = kwargs[:maxIter]
    end
    if haskey(kwargs, :minStepFac)
        Base.depwarn("'minStepFac' argument is deprecated, use 'minstepfac' instead", :fit!)
        minstepfac = kwargs[:minStepFac]
    end
    if haskey(kwargs, :convTol)
        Base.depwarn("'convTol' argument is deprecated, use 'tol' instead", :fit!)
        tol = kwargs[:convTol]
    end
    if !issubset(keys(kwargs), (:maxIter, :minStepFac, :convTol))
        throw(ArgumentError("unsupported keyword argument"))
    end

    maxiter >= 1 || throw(ArgumentError("maxiter must be positive"))
    tol > 0  || throw(ArgumentError("tol must be positive"))
    initialθ > 0 || throw(ArgumentError("initialθ must be positive"))

    # fit a Poisson regression model if the user does not specify an initial θ
    if isinf(initialθ) 
        regmodel = glm(F, D, Poisson(), args...;
                       maxiter=maxiter, tol=tol, verbose=verbose, kwargs...)
    else
        regmodel = glm(F, D, NegativeBinomial(initialθ), args...;
                       maxiter=maxiter, tol=tol, verbose=verbose, kwargs...)
    end

    μ = regmodel.model.rr.mu
    y = regmodel.model.rr.y
    wts = regmodel.model.rr.wts
    lw, ly = length(wts), length(y)
    if lw != ly && lw != 0
        throw(ArgumentError("length of wts must be either $ly or 0 but was $lw"))
    end

    θ = mle_for_θ(y, μ, wts)
    d = sqrt(2 * max(1, deviance(regmodel)))
    δ = one(θ)
    ll = loglikelihood(regmodel)
    ll0 = ll + 2 * d

    converged = false
    for i = 1:maxiter
        if abs(ll0 - ll)/d + abs(δ) <= tol
            converged = true
            break
        end
        verbose && println("[ Alternating iteration ", i, ", θ = ", θ, " ]")
        regmodel = glm(F, D, NegativeBinomial(θ), args...;
                       maxiter=maxiter, tol=tol, verbose=verbose, kwargs...)
        μ = regmodel.model.rr.mu
        prevθ = θ
        θ = mle_for_θ(y, μ, wts; maxiter=maxiter, tol=tol)
        δ = prevθ - θ
        ll0 = ll
        ll = loglikelihood(regmodel)
    end
    converged || throw(ConvergenceException(maxiter))
    regmodel
end
