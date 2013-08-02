using DataFrames, Distributions, GLM  # should be externally available
module MixedModels

    using DataFrames, Distributions, NLopt, NumericExtensions
    using Base.LinAlg.BLAS: gemm!, gemv!, syrk!, syrk, trmm!, trmm, trmv!, trsm!, trsv!
    using Base.LinAlg.CHOLMOD: CholmodDense, CholmodDense!, CholmodFactor,
          CholmodSparse, CholmodSparse!, chm_scale!, CHOLMOD_L, CHOLMOD_Lt, CHOLMOD_P, CHOLMOD_Pt
    using Base.LinAlg.LAPACK:  potrf!, potrs!

    import Base: cor, cholfact, logdet, scale, show, size, solve, std
    import Distributions: fit
    import GLM: coef, coeftable, confint, deviance, df_residual, linpred, stderr, vcov

    export                              # types
        MixedModel,
        LinearMixedModel,
        LMMGeneral,
        LMMScalar1,
                                        # functions
        fixef,
        grplevels,
        isfit,
        lmer,
        lower,
        objective,
        pwrss,
        ranef,
        reml!,
        reml,
        solve!,
        theta!,
        theta

#    include("rsc.jl")            # regular sparse column-oriented matrices
    include("utils.jl")         # utilities to deal with the model formula
    include("LinearMixedModels.jl")     # method definitions for the abstract class
    include("LMMGeneral.jl")    # general form of linear mixed-effects models
    include("LMMScalar1.jl")    # models with a single, scalar random-effects term
    include("lmer.jl")          # fit and analyze linear mixed-effects models
#    include("vectorlmm.jl")


end #module

