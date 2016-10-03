__precompile__()

module GLM
    using Reexport
    @reexport using Distributions
    using Base.LinAlg.LAPACK: potrf!, potrs!
    using Base.LinAlg.BLAS: gemm!, gemv!
    using Base.LinAlg: QRCompactWY, Cholesky
    using StatsBase: StatsBase, CoefTable, StatisticalModel, RegressionModel
    using StatsFuns: logit, logistic
    using Distributions: sqrt2, sqrt2π

    import Base: (\), cholfact, convert, cor, show, size
    import StatsBase: coef, coeftable, confint, deviance, nulldeviance, df, df_residual, loglikelihood, nullloglikelihood, nobs, stderr, vcov, residuals, predict, fit, model_response, r2, r², adjr2, adjr²
    import StatsFuns: xlogy
    export coef, coeftable, confint, deviance, nulldeviance, df, df_residual, loglikelihood, nobs, stderr, vcov, residuals, predict, fit, fit!, model_response, r2, r², adjr2, adjr²

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
        lm,             # linear model (QR factorization)
        lmc,            # linear model (Cholesky factorization)
        mueta,          # derivative of inverse link
        mustart,        # derive starting values for the mu vector
        nobs,           # total number of observations
        predict,        # make predictions
        updateμ!,      # update the response type from the linear predictor
        wrkresp         # working response

    typealias FP AbstractFloat
    typealias FPVector{T<:FP} DenseArray{T,1}

    abstract ModResp                   # model response

    abstract LinPred                   # linear predictor in statistical models
    abstract DensePred <: LinPred      # linear predictor with dense X
    abstract LinPredModel <: RegressionModel # model based on a linear predictor

    include("linpred.jl")
    include("lm.jl")
    include("glmtools.jl")
    include("glmfit.jl")

end # module
