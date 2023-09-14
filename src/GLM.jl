module GLM
    using Distributions, LinearAlgebra, Printf, Reexport, SparseArrays, Statistics, StatsBase, StatsFuns
    using LinearAlgebra: copytri!, QRCompactWY, Cholesky, CholeskyPivoted, BlasReal
    using Printf: @sprintf
    using StatsBase: CoefTable, StatisticalModel, RegressionModel
    using StatsFuns: logit, logistic
    @reexport using StatsModels
    using Distributions: sqrt2, sqrt2π

    import Base: (\), convert, show, size
    import LinearAlgebra: cholesky, cholesky!
    import Statistics: cor
    using StatsAPI
    import StatsBase: coef, coeftable, coefnames, confint, deviance, nulldeviance, dof, dof_residual,
                      loglikelihood, nullloglikelihood, nobs, stderror, vcov,
                      residuals, predict, predict!,
                      fitted, fit, model_response, response, modelmatrix, r2, r², adjr2, adjr², PValue
    import StatsFuns: xlogy
    import SpecialFunctions: erfc, erfcinv, digamma, trigamma
    import StatsModels: hasintercept
    export coef, coeftable, confint, deviance, nulldeviance, dof, dof_residual,
           loglikelihood, nullloglikelihood, nobs, stderror, vcov, residuals, predict,
           fitted, fit, fit!, model_response, response, modelmatrix, r2, r², adjr2, adjr²,
           cooksdistance, hasintercept, dispersion, vif, gvif, termnames

    export
        # types
        ## Distributions
        Bernoulli,
        Binomial,
        Gamma,
        Geometric,
        InverseGaussian,
        NegativeBinomial,
        Normal,
        Poisson,

        ## Link types
        Link,
        CauchitLink,
        CloglogLink,
        IdentityLink,
        InverseLink,
        InverseSquareLink,
        LogitLink,
        LogLink,
        NegativeBinomialLink,
        PowerLink,
        ProbitLink,
        SqrtLink,

        # Model types
        GeneralizedLinearModel,
        LinearModel,

        # functions
        canonicallink,  # canonical link function for a distribution
        deviance,       # deviance of fitted and observed responses
        devresid,       # vector of squared deviance residuals
        formula,        # extract the formula from a model
        glm,            # general interface
        linpred,        # linear predictor
        lm,             # linear model
        negbin,         # interface to fitting negative binomial regression
        nobs,           # total number of observations
        predict,        # make predictions
        ftest           # compare models with an F test

    const FP = AbstractFloat
    const FPVector{T<:FP} = AbstractArray{T,1}

    """
        ModResp

    Abstract type representing a model response vector
    """
    abstract type ModResp end                         # model response

    """
        LinPred

    Abstract type representing a linear predictor
    """
    abstract type LinPred end                         # linear predictor in statistical models
    abstract type DensePred <: LinPred end            # linear predictor with dense X
    abstract type LinPredModel <: RegressionModel end # model based on a linear predictor

    @static if VERSION < v"1.8.0-DEV.1139"
        pivoted_cholesky!(A; kwargs...) = cholesky!(A, Val(true); kwargs...)
    else
        pivoted_cholesky!(A; kwargs...) = cholesky!(A, RowMaximum(); kwargs...)
    end

    include("linpred.jl")
    include("lm.jl")
    include("glmtools.jl")
    include("glmfit.jl")
    include("ftest.jl")
    include("negbinfit.jl")
    include("deprecated.jl")

end # module
