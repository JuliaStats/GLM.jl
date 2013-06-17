using DataFrames, Distributions         # should be externally available
module MixedModels

    using DataFrames, Distributions, NLopt
    using Base.LinAlg.BLAS: gemm!, gemv!, syrk!, syrk, trmm!, trmm, trmv!, trsm!, trsv!
    using Base.LinAlg.CHOLMOD: CholmodDense, CholmodDense!, CholmodFactor,
          CholmodSparse, CholmodSparse!, chm_scale!, CHOLMOD_L, CHOLMOD_Lt, CHOLMOD_P, CHOLMOD_Pt
    using Base.LinAlg.LAPACK:  potrf!, potrs!
    using Base.LinAlg.UMFPACK: increment!, increment

    import Base: (*), A_mul_Bc, Ac_mul_B, AbstractSparseMatrix, copy, dense, fill!,
                 full, logdet, nnz, show, size, solve, sparse
    import Base.LinAlg.CHOLMOD.CholmodSparse
    import Distributions: deviance, fit

    export                              # types
        MixedModel,
        SparseMatrixRSC,
        LMMGeneral,
        LMMsimple,
        LMMsplit,
        ScalarLMM1,
        VectorLMM1,
                                        # functions
        fixef!,
        lmer,
        updateL!,
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

