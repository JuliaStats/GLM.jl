module MixedModels

using DataFrames, Distributions, NLopt
using Base.LinAlg.CHOLMOD.CholmodDense
using Base.LinAlg.CHOLMOD.CholmodFactor
using Base.LinAlg.CHOLMOD.CholmodSparse
using Base.LinAlg.CHOLMOD.chm_scale!
using Base.LinAlg.CHOLMOD.chm_factorize!

import Base.(*), Base.A_mul_Bc, Base.Ac_mul_B, Base.AbstractSparseMatrix
import Base.copy, Base.dense, Base.fill!
import Base.full, Base.nnz, Base.show, Base.size, Base.sparse
import Base.LinAlg.CHOLMOD.CholmodSparse
import Distributions.deviance, Distributions.fit

export                                  # types
    MixedModel,
    SparseMatrixRSC,
    LMMsimple,
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

