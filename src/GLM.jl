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
    import StatsBase: coef, coeftable, confint, deviance, nulldeviance, dof, dof_residual,
                      loglikelihood, nullloglikelihood, nobs, stderror, vcov, residuals, predict,
                      fit, model_response, r2, r², adjr2, adjr², PValue
    import StatsFuns: xlogy
    import SpecialFunctions: erfc, erfcinv, digamma, trigamma
    export coef, coeftable, confint, deviance, nulldeviance, dof, dof_residual,
           loglikelihood, nullloglikelihood, nobs, stderror, vcov, residuals, predict,
           fit, fit!, model_response, r2, r², adjr2, adjr²

    export                              # types
        Bernoulli,
        Binomial,
        CauchitLink,
        CloglogLink,
        DensePred,
        DensePredQR,
        DensePredChol,
        Gamma,
        Gaussian,
        GeneralizedLinearModel,
        GlmResp,
        IdentityLink,
        InverseGaussian,
        InverseLink,
        InverseSquareLink,
        LinearModel,
        Link,
        LinPred,
        LinPredModel,
        LogitLink,
        LogLink,
        LmResp,
        NegativeBinomial,
        NegativeBinomialLink,
        Normal,
        Poisson,
        ProbitLink,
        SqrtLink,

                                        # functions
        canonicallink,  # canonical link function for a distribution
        delbeta!,       # evaluate the increment in the coefficient vector
        deviance,       # deviance of fitted and observed responses
        devresid,       # vector of squared deviance residuals
        formula,        # extract the formula from a model
        glm,            # general interface
        glmvar,         # the variance function
        inverselink,    # returns μ, dμ/dη and, when appropriate, μ*(1-μ)
        linkfun,        # link function mapping mu to eta, the linear predictor
        linkinv,        # inverse link mapping eta to mu
        linpred,        # linear predictor
        linpred!,       # update the linear predictor in place
        lm,             # linear model
        logistic,
        logit,
        mueta,          # derivative of inverse link
        mustart,        # derive starting values for the mu vector
        negbin,         # interface to fitting genative binomial regression
        nobs,           # total number of observations
        predict,        # make predictions
        updateμ!,       # update the response type from the linear predictor
        wrkresp,        # working response
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

    include("linpred.jl")
    include("lm.jl")
    include("glmtools.jl")
    include("glmfit.jl")
    include("ftest.jl")
    include("negbinfit.jl")
    include("deprecated.jl")

end # module
