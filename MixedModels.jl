using DataFrames, Distributions, GLM  # should be externally available
module MixedModels

    using DataFrames, Distributions, NLopt
    using Base.LinAlg.BLAS: gemm!, gemv!, syrk!, syrk, trmm!, trmm, trmv!, trsm!, trsv!
    using Base.LinAlg.CHOLMOD: CholmodDense, CholmodDense!, CholmodFactor,
          CholmodSparse, CholmodSparse!, chm_scale!, CHOLMOD_L, CHOLMOD_Lt, CHOLMOD_P, CHOLMOD_Pt
    using Base.LinAlg.LAPACK:  potrf!, potrs!

    import Base: show, size, solve
    import Distributions: deviance, fit
    import GLM: df_residual, stderr, vcov

    export                              # types
        MixedModel,
        LinearMixedModel,
        LMMGeneral,
                                        # functions
        VarCorr,
        fixef,
        grplevels,
        isfit,
        isreml,
        lmer,
        lower,
        objective,
        pwrss,
        ranef,
        solve!,
        theta!,
        theta

    typealias VTypes Union(Float64,Complex128)
    typealias ITypes Union(Int32,Int64)

    abstract MixedModel

#    include("rsc.jl")            # regular sparse column-oriented matrices
#    include("utils.jl")         # utilities to deal with the model formula
    include("linearmixedmodel.jl")
#    include("scalarlmm.jl")
#    include("vectorlmm.jl")


end #module

