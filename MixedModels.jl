using DataFrames, Distributions         # should be externally available
module MixedModels

    using DataFrames, Distributions, NLopt
    using Base.LinAlg.BLAS:    gemv!, syrk!, syrk
    using Base.LinAlg.CHOLMOD: CholmodDense, CholmodDense!,
      CholmodFactor, CholmodSparse, chm_scale!, chm_factorize!,
      chm_factorize_p!, CHOLMOD_L, CHOLMOD_Lt
    using Base.LinAlg.LAPACK:  potrf!, potrs!
    using Base.LinAlg.UMFPACK: increment!, increment

    import Base: (*), A_mul_Bc, Ac_mul_B, AbstractSparseMatrix, copy, dense, fill!,
                 full, nnz, show, size, sparse
    import Base.LinAlg.CHOLMOD.CholmodSparse
    import Distributions: deviance, fit

    export                              # types
        MixedModel,
        SparseMatrixRSC,
        LMMsimple,
        LMMsplit,
        ScalarLMM1,
        VectorLMM1,
                                        # functions
        fixef!,
        lmer,
        objective!,
        ranef!,
        reml!,
        VarCorr!

    typealias VTypes Union(Float64,Complex128)
    typealias ITypes Union(Int32,Int64)

    abstract MixedModel

    include("rsc.jl")            # regular sparse column-oriented matrices
    include("utils.jl")         # utilities to deal with the model formula
    include("linearmixedmodel.jl")
    include("scalarlmm.jl")
    include("vectorlmm.jl")


end #module

