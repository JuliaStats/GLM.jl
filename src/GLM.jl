using DataFrames, Distributions, NumericExtensions

module GLM

    using DataFrames, Distributions, NLopt, NumericExtensions
    using Base.LinAlg.LAPACK: geqrt3!, potrf!, potri!, potrs!
    using Base.LinAlg.BLAS: gemm!, gemv!, symmetrize!, syrk!, syrk, trmm!, trmm, trmv!, trsm!, trsv!
    using Base.LinAlg.CHOLMOD: CholmodDense, CholmodDense!, CholmodFactor,
          CholmodSparse, CholmodSparse!, chm_scale!, CHOLMOD_L, CHOLMOD_Lt, CHOLMOD_P, CHOLMOD_Pt

    import Base: (\), cholfact, cor, logdet, scale, show, size, solve
    import Distributions: fit, logpdf
    import DataFrames: ModelFrame, ModelMatrix, model_response
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
        LinearMixedModel,
        LMMGeneral,
        LMMScalar1,
        LogitLink,
        LogLink,
        LmMod,
        LmResp,
        MixedModel,
        ProbitLink,
                                        # functions
        canonicallink,  # canonical link function for a distribution
        coef,           # estimated coefficients
        coeftable,      # coefficients, standard errors, etc.
        confint,        # confidence intervals on coefficients
        contr_treatment,# treatment contrasts
        delbeta!,       # evaluate the increment in the coefficient vector
        deviance,       # deviance of fitted and observed responses
        devresid,       # vector of squared deviance residuals
        df_residual,    # degrees of freedom for residuals
        drsum,          # sum of squared deviance residuals
        fixef,          # extract the fixed-effects parameter estimates
        formula,        # extract the formula from a model
        glm,            # general interface
        grplevels,      # number of levels per grouping factor in mixed-effects models
        isfit,          # predictate to check if a model has been fit
        linkfun!,       # mutating link function
        linkfun,        # link function mapping mu to eta, the linear predictor
        linkinv!,       # mutating inverse link
        linkinv,        # inverse link mapping eta to mu
        linpred,        # linear predictor
        lm,             # linear model (QR factorization)
        lmc,            # linear model (Cholesky factorization)          
        lmm,            # fit a linear mixed-effects model (LMM)
        lower,          # vector of lower bounds on parameters in mixed-effects models
        mueta!,         # mutating derivative of inverse link
        mueta,          # derivative of inverse link
        mustart,        # derive starting values for the mu vector
        nobs,           # total number of observations
        objective,      # the objective function in fitting a model
        predict,        # make predictions
        pwrss,          # penalized, weighted residual sum-of-squares
        ranef,          # extract the conditional modes of the random effects
        reml!,          # set the objective to be the REML criterion
        reml,           # is the objective the REML criterion?
        residuals,      # extractor for residuals
        solve!,         # update the coefficients by solving the MME's
        sqrtwrkwt,      # square root of the working weights
        stderr,         # standard errors of the coefficients
        theta!,         # set the value of the variance component parameters        
        theta,          # extract the variance-component parameter vector
        updatemu!,      # update the response type from the linear predictor
        var!,           # mutating variance function
        vcov,           # estimated variance-covariance matrix of coef
        wrkresid!,      # mutating working residuals function
        wrkresid,       # extract the working residuals              
        wrkresp         # working response

    typealias FP FloatingPoint

    abstract ModResp                   # model response

    abstract LinPred                   # linear predictor in statistical models
    abstract DensePred <: LinPred      # linear predictor with dense X
    abstract LinPredModel              # model based on a linear predictor

    abstract MixedModel                # model with fixed and random effects
    abstract LinearMixedModel <: MixedModel # Gaussian mixed model with identity link

    typealias VTypes Union(Float64,Complex128)
    typealias ITypes Union(Int32,Int64)

    include("linpred.jl")
    include("lm.jl")
    include("glmtools.jl")
    include("glmfit.jl")

    include("utils.jl")     # utilities to deal with the model formula
    include("LinearMixedModels.jl") # method definitions for the abstract class
    include("LMMGeneral.jl") # general form of linear mixed-effects models
    include("LMMScalar1.jl") # models with a single, scalar random-effects term
    include("lmm.jl")    # fit and analyze linear mixed-effects models

end # module
