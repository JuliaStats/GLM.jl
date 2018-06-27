function mle_for_θ(y::AbstractVector, μ::AbstractVector, wts::AbstractVector;
                 maxIter = 30, convTol = 1.e-6)
    function firstDeriv(θ::Real)
        tmp(y, θ, μ) = (y+θ)./(μ+θ) + log.(μ+θ) - 1 - log.(θ) - digamma.(θ+y) + digamma(θ)
        unitWeights ? sum(tmp(y, θ, μ)) : sum(wts .* tmp(y, θ, μ))
    end
    function secondDeriv(θ::Real)
        tmp(y, θ, μ) = -(y+θ)./(μ+θ).^2 + 2 ./ (μ+θ) - 1/θ - trigamma.(θ+y) + trigamma(θ)
        unitWeights ? sum(tmp(y, θ, μ)) : sum(wts .* tmp(y, θ, μ))
    end

    unitWeights = length(wts) == 0
    n = unitWeights ? length(y) : sum(wts)
    θ = n / (unitWeights ? sum((y ./ μ - 1).^2) : sum(wts .* (y ./ μ - 1).^2))
    δ, converged = one(θ), false
    for t = 1:maxIter
        θ = abs(θ)
        δ = firstDeriv(θ) / secondDeriv(θ)
        if (abs(δ) <= convTol)
            converged = true
            break
        end
        θ = θ - δ
    end
    converged || throw(ConvergenceException(maxIter))
    θ
end

function negbin(F, D, args...; initialθ::Real=Inf, maxIter::Integer=30, convTol::Real=1.e-6, verbose::Bool=false, kwargs...)
    maxIter >= 1 || throw(ArgumentError("maxIter must be positive"))
    convTol > 0  || throw(ArgumentError("convTol must be positive"))
    initialθ > 0 || throw(ArgumentError("initialθ must be positive"))

    # fit a Poisson if the user does not specify an initial θ
    regModel = isinf(initialθ) ? glm(F, D, Poisson(), args...; maxIter=maxIter, convTol=convTol, verbose=verbose, kwargs...) :
                                 glm(F, D, NegativeBinomial(initialθ), args...; maxIter=maxIter, convTol=convTol, verbose=verbose, kwargs...)

    μ = regModel.model.rr.mu
    y = regModel.model.rr.y
    wts = regModel.model.rr.wts
    lw, ly = length(wts), length(y)
    if lw != ly && lw != 0
        throw(ArgumentError("length of wts must be either $ly or 0 but was $lw"))
    end

    θ = mle_for_θ(y, μ, wts)
    d = sqrt(2 * max(1, deviance(regModel)))
    δ = one(θ)
    ll = loglikelihood(regModel)
    ll0 = ll + 2 * d

    converged = false
    for i = 1:maxIter
        if (abs(ll0 - ll)/d + abs(δ) <= convTol)
            converged = true
            break
        end
        verbose && println("[ Alternating iteration ", i, ", θ = ", θ, "]")
        regModel = glm(F, D, NegativeBinomial(θ), args...; maxIter=maxIter, convTol=convTol, verbose=verbose, kwargs...)
        μ = regModel.model.rr.mu
        prevθ = θ
        θ = mle_for_θ(y, μ, wts; maxIter = maxIter, convTol = convTol)
        δ = prevθ - θ
        ll0 = ll
        ll = loglikelihood(regModel)
    end
    converged || throw(ConvergenceException(maxIter))
    regModel
end
