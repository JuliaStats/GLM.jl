mutable struct NegBinRegressionModel{G<:GlmResp,L<:LinPred} <: AbstractGLM
    rr::G
    pp::L
    fit::Bool
    θ::Real
end

function mleForθ(y::AbstractVector, μ::AbstractVector, wts::AbstractVector;
                 maxIter = 30, convTol = 1.e-6)
    function firstDeriv(θ::Real)
        tmp = (y+θ)./(μ+θ) + log.(μ+θ) - 1 - log.(θ) - digamma.(θ+y) + digamma(θ)
        unitWeights ? sum(tmp) : sum(wts .* tmp)
    end
    function secondDeriv(θ::Real)
        tmp = -(y+θ)./(μ+θ).^2 + 2 ./ (μ+θ) - 1/θ - trigamma.(θ+y) + trigamma(θ)
        unitWeights ? sum(tmp) : sum(wts .* tmp)
    end

    unitWeights = length(wts) == 0
    n = unitWeights ? length(y) : sum(wts)
    θ = n / (unitWeights ? sum((y ./ μ - 1).^2) : sum(wts .* (y ./ μ - 1).^2))
    δ, converged = 1, false
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

function negbin(F, D, args...; kwargs...)
    # if initialθ was specified, then use it to train the initial model, remove the argument from fitargs;
    # otherwise, the initial model is trained with Poisson regression
    new_kwargs = Dict(kwargs)
    initialθset = haskey(new_kwargs, :initialθ)
    maxIter = haskey(new_kwargs, :maxIter) ? new_kwargs[:maxIter] : 30
    convTol = haskey(new_kwargs, :convTol) ? new_kwargs[:convTol] : 1.e-6
    verbose = haskey(new_kwargs, :verbose) ? new_kwargs[:verbose] : false

    initialθ = 1.0
    if initialθset
        initialθ = new_kwargs[:initialθ]
        delete!(new_kwargs, :initialθ)
    end
    maxIter >= 1 || throw(ArgumentError("maxIter must be positive"))
    convTol > 0  || throw(ArgumentError("convTol must be positive"))
    initialθ > 0 || throw(ArgumentError("initialθ must be positive"))

    regModel = initialθset ? glm(F, D, NegativeBinomial(initialθ), args...; new_kwargs...) :
                             glm(F, D, Poisson(), args...; new_kwargs...)

    μ = regModel.model.rr.mu
    y = regModel.model.rr.y
    wts = regModel.model.rr.wts
    lw, ly = length(wts), length(y)
    if (lw != ly && lw != 0)
        throw(ArgumentError("length of wts must be either $ly or 0 but was $lw"))
    end

    println("Max iterations = ", maxIter)
    println("convergence tolerance = ", convTol)
    println("weights = ", wts)

    θ = mleForθ(y, μ, wts)
    d = sqrt(2 * max(1, deviance(regModel))); println("d = ", d)
    δ = 1
    lm = loglikelihood(regModel)
    lm0 = lm + 2 * d

    converged = false
    for i = 1:maxIter
        if (abs(lm0 - lm)/d + abs(δ) <= convTol)
            converged = true
            break
        end
        verbose && println("[ Alternating iteration ", i, ", θ = ", θ, "]")
        regModel = glm(F, D, NegativeBinomial(θ), args...; new_kwargs...)
        μ = regModel.model.rr.mu
        prevθ = θ
        θ = mleForθ(y, μ, wts; maxIter = maxIter, convTol = convTol)
        δ = prevθ - θ
        lm0 = lm
        lm = loglikelihood(regModel)
    end
    converged || throw(ConvergenceException(maxIter))
    regModel
end
