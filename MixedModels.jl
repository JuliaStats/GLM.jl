using DataFrames, Distributions         # should be externally available
module MixedModels

using DataFrames, Distributions, NLopt
using Base.LinAlg.BLAS.syrk!
using Base.LinAlg.CHOLMOD.CholmodDense
using Base.LinAlg.CHOLMOD.CholmodDense!
using Base.LinAlg.CHOLMOD.CholmodFactor
using Base.LinAlg.CHOLMOD.CholmodSparse
using Base.LinAlg.CHOLMOD.chm_scale!
using Base.LinAlg.CHOLMOD.chm_factorize!
using Base.LinAlg.CHOLMOD.chm_factorize_p!
using Base.LinAlg.CHOLMOD.CHOLMOD_L
using Base.LinAlg.CHOLMOD.CHOLMOD_Lt
using Base.LinAlg.LAPACK.potrf!
using Base.LinAlg.UMFPACK.increment!
using Base.LinAlg.UMFPACK.increment

import Base: (*), A_mul_Bc, Ac_mul_B, AbstractSparseMatrix, copy, dense, fill!,
             full, nnz, show, size, sparse
import Base.LinAlg.CHOLMOD.CholmodSparse
import Distributions: deviance, fit

export                                  # types
    MixedModel,
    SparseMatrixRSC,
    LMMsimple,
    LMMsplit,
                                        # functions
    fixef,
    ranef,
    reml,
    VarCorr

typealias VTypes Union(Float64,Complex128)
typealias ITypes Union(Int32,Int64)

abstract MixedModel

include("RSC.jl")            # regular sparse column-oriented matrices
include("utils.jl")         # utilities to deal with the model formula
include("linearmixedmodel.jl")
include("LMM.jl")

end #module

