__precompile__()

module GLM
    using Reexport
    @reexport using Distributions
    using Base.LinAlg.LAPACK: potrf!, potrs!
    using Base.LinAlg.BLAS: gemm!, gemv!
    using Base.LinAlg: QRCompactWY, Cholesky
    using StatsBase: StatsBase, CoefTable, StatisticalModel, RegressionModel
    using Distributions: sqrt2, sqrt2π
    using Compat

    import Base: (\), cholfact, convert, cor, show, size
    import StatsBase: coef, coeftable, confint, deviance, nulldeviance, dof, dof_residual, loglikelihood, nullloglikelihood, nobs, stderr, vcov, residuals, predict, fit, model_response, r2, r², adjr2, adjr², PValue
    import SpecialFunctions: erfc, erfcinv
    export coef, coeftable, confint, deviance, nulldeviance, dof, dof_residual, loglikelihood, nobs, stderr, vcov, residuals, predict, fit, fit!, model_response, r2, r², adjr2, adjr²

    export                              # types
        CauchitLink,
        CloglogLink,
        DensePred,
        DensePredQR,
        DensePredChol,
        GeneralizedLinearModel,
        GlmResp,
        IdentityLink,
        InverseLink,
        LinearModel,
        Link,
        LinPred,
        LinPredModel,
        LogitLink,
        LogLink,
        LmResp,
        ProbitLink,
        SqrtLink,

                                        # functions
        canonicallink,  # canonical link function for a distribution
        delbeta!,       # evaluate the increment in the coefficient vector
        deviance,       # deviance of fitted and observed responses
        devresid,       # vector of squared deviance residuals
        formula,        # extract the formula from a model
        glm,            # general interface
        linkfun,        # link function mapping mu to eta, the linear predictor
        linkinv,        # inverse link mapping eta to mu
        linpred,        # linear predictor
        linpred!,       # update the linear predictor
        lm,             # linear model
        mueta,          # derivative of inverse link
        mustart,        # derive starting values for the mu vector
        nobs,           # total number of observations
        predict,        # make predictions
        updateμ!,      # update the response type from the linear predictor
        wrkresp,        # working response
        ftest           # compare models with an F test

    const FP = AbstractFloat
    @compat const FPVector{T<:FP} = AbstractArray{T,1}

    @compat abstract type ModResp end                         # model response

    @compat abstract type LinPred end                         # linear predictor in statistical models
    @compat abstract type DensePred <: LinPred end            # linear predictor with dense X
    @compat abstract type LinPredModel <: RegressionModel end # model based on a linear predictor

    include("linpred.jl")
    include("lm.jl")
    include("glmtools.jl")
    include("glmfit.jl")
    include("ftest.jl")

end # module
