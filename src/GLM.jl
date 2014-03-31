using Distributions, NumericExtensions

module GLM

    using Distributions, NumericExtensions, NumericFuns
    using Base.LinAlg.LAPACK: potrf!, potrs!
    using Base.LinAlg.BLAS: gemm!, gemv!
    using Base.LinAlg: QRCompactWY
    using StatsBase: CoefTable, StatisticalModel, RegressionModel

    import Base: (\), cholfact, cor, scale, show, size
    import StatsBase
    import StatsBase: coef, coeftable, confint, deviance, loglikelihood, nobs, stderr,
                      vcov, residuals, predict, fit
    import NumericExtensions: evaluate, result_type

    export                              # types
        CauchitLink,
        CloglogLink,
        DensePred,
        DensePredQR,
        DensePredChol,
        GlmMod,
        GlmResp,
        IdentityLink,
        InverseLink,
        Link,
        LinPred,
        LinPredModel,
        LogitLink,
        LogLink,
        LmMod,
        LmResp,
        ProbitLink,
                                        # functions
        canonicallink,  # canonical link function for a distribution
        contr_treatment,# treatment contrasts
        delbeta!,       # evaluate the increment in the coefficient vector
        deviance,       # deviance of fitted and observed responses
        devresid,       # vector of squared deviance residuals
        df_residual,    # degrees of freedom for residuals
        drsum,          # sum of squared deviance residuals
        fit,            # function to fit models, from StatsBase
        formula,        # extract the formula from a model
        glm,            # general interface
        linkfun!,       # mutating link function
        linkfun,        # link function mapping mu to eta, the linear predictor
        linkinv!,       # mutating inverse link
        linkinv,        # inverse link mapping eta to mu
        linpred,        # linear predictor
        linpred!,       # update the linear predictor
        lm,             # linear model (QR factorization)
        lmc,            # linear model (Cholesky factorization)          
        mueta!,         # mutating derivative of inverse link
        mueta,          # derivative of inverse link
        mustart,        # derive starting values for the mu vector
        nobs,           # total number of observations
        objective,      # the objective function in fitting a model
        predict,        # make predictions
        sqrtwrkwt,      # square root of the working weights
        stderr,         # standard errors of the coefficients
        updatemu!,      # update the response type from the linear predictor
        var!,           # mutating variance function
        wrkresid!,      # mutating working residuals function
        wrkresid,       # extract the working residuals              
        wrkresp         # working response

    typealias FP FloatingPoint
    typealias FPStoredVector{T<:FloatingPoint} StoredArray{T,1}

    abstract ModResp                   # model response

    abstract LinPred                   # linear predictor in statistical models
    abstract DensePred <: LinPred      # linear predictor with dense X
    abstract LinPredModel <: RegressionModel # model based on a linear predictor

    include("linpred.jl")
    include("lm.jl")
    include("glmtools.jl")
    include("glmfit.jl")
    include("deprecated.jl")

end # module
