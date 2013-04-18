module MixedModels

using DataFrames, Distributions, NLopt
using Base.LinAlg.CHOLMOD.CholmodDense
using Base.LinAlg.CHOLMOD.CholmodFactor
using Base.LinAlg.CHOLMOD.CholmodSparse
using Base.LinAlg.CHOLMOD.chm_scale!
using Base.LinAlg.CHOLMOD.chm_factorize!

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

include("RSC.jl")                       # regular sparse column-oriented matrices

abstract MixedModel
abstract LinearMixedModel <: MixedModel

include("LMM.jl")

end #module

