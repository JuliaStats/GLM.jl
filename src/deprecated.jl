import Base

@deprecate(fit(m::AbstractGLM; verbose::Bool=false, maxIter::Integer=30,
               minStepFac::Real=0.001, convTol::Real=1.e-6, start=nothing),
           fit!(m; verbose=verbose, maxIter=maxIter, minStepFac=minStepFac,
                convTol=convTol, start=start))

@deprecate(fit(m::AbstractGLM, y; wts=nothing, offset=nothing, dofit::Bool=true,
               verbose::Bool=false, maxIter::Integer=30, minStepFac::Real=0.001, convTol::Real=1.e-6,
               start=nothing),
           fit!(m, y; wts=wts, offset=offset, dofit=dofit,
                verbose=verbose, maxIter=maxIter, minStepFac=minStepFac, convTol=convTol,
                start=start))

@deprecate lmc(X, y) fit(LinearModel{Chol}, X, y)

typealias LmMod LinearModel
typealias GlmMod GeneralizedLinearModel
typealias DensePredQR DenseQRUnweighted
typealias DensePredChol DenseCholUnweighted
