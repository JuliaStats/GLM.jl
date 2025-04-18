module GLM
using Distributions, LinearAlgebra, Printf, Reexport, Statistics, StatsBase
using LinearAlgebra: copytri!, QRCompactWY, Cholesky, CholeskyPivoted, BlasReal
using Printf: @sprintf
using StatsBase: CoefTable, StatisticalModel, RegressionModel
using LogExpFunctions: logistic, logit, xlogy
@reexport using StatsModels
using Distributions: sqrt2, sqrt2π

import Base: (\), convert, show, size
import LinearAlgebra: cholesky, cholesky!
import Statistics: cor
using StatsAPI
import StatsBase: coef, coeftable, coefnames, confint, deviance, nulldeviance, dof,
                  dof_residual,
                  leverage, loglikelihood, nullloglikelihood, nobs, stderror, vcov,
                  residuals, predict, predict!,
                  fitted, fit, model_response, response, modelmatrix, r2, r², adjr2, adjr²,
                  PValue,
                  fweights, pweights, aweights
import SpecialFunctions: erfc, erfcinv, digamma, trigamma
import StatsModels: hasintercept
using Tables: Tables
export coef, coeftable, confint, deviance, nulldeviance, dof, dof_residual,
       loglikelihood, nullloglikelihood, nobs, stderror, vcov, residuals, predict,
       fitted, fit, fit!, model_response, response, modelmatrix, r2, r², adjr2, adjr²,
       cooksdistance, hasintercept, dispersion, vif, gvif, termnames

export
# types
## Distributions
      Bernoulli,
      Binomial,
      Gamma,
      Geometric,
      InverseGaussian,
      NegativeBinomial,
      Normal,
      Poisson,

## Link types
      Link,
      CauchitLink,
      CloglogLink,
      IdentityLink,
      InverseLink,
      InverseSquareLink,
      LogitLink,
      LogLink,
      NegativeBinomialLink,
      PowerLink,
      ProbitLink,
      SqrtLink,

# Model types
      GeneralizedLinearModel,
      LinearModel,

# functions
      canonicallink,  # canonical link function for a distribution
      deviance,       # deviance of fitted and observed responses
      devresid,       # vector of squared deviance residuals
      formula,        # extract the formula from a model
      glm,            # general interface
      leverage,       # leverage 
      linpred,        # linear predictor
      lm,             # linear model
      negbin,         # interface to fitting negative binomial regression
      nobs,           # total number of observations
      predict,        # make predictions
      ftest,          # compare models with an F test
      fweights,
      pweights,
      aweights

const FP = AbstractFloat
const FPVector{T<:FP} = AbstractArray{T,1}

"""
    ModResp

Abstract type representing a model response vector
"""
abstract type ModResp end                         # model response

"""
    LinPred

Abstract type representing a linear predictor
"""
abstract type LinPred end                         # linear predictor in statistical models
abstract type DensePred <: LinPred end            # linear predictor with dense X
abstract type LinPredModel <: RegressionModel end # model based on a linear predictor

const COMMON_FIT_KWARGS_DOCS = """
    - `dropcollinear::Bool`: Controls whether or not a model matrix
      less-than-full rank is accepted.
      If `true` (the default) the coefficient for redundant linearly dependent columns is
      `0.0` and all associated statistics are set to `NaN`.
      Typically from a set of linearly-dependent columns the last ones are identified as redundant
      (however, the exact selection of columns identified as redundant is not guaranteed).
    - `method::Symbol`: Controls which decomposition method to use.
      If `method=:qr` (the default), then the `QR` decomposition method will be used.
      If `method=:cholesky`, then the `Cholesky` decomposition method will be used.
      The Cholesky decomposition is faster and more computationally efficient than
      QR, but is less numerically stable and thus may fail or produce less accurate
      estimates for some models.
    - `wts::AbstractWeights`: Weights of observations.
       The weights can be of type `AnalyticWeights`, `FrequencyWeights`, 
       `ProbabilityWeights`, or `UnitWeights`. `AnalyticWeights` describe a non-random
       relative importance (usually between 0 and 1) for each observation. These weights may 
       also be referred to as reliability weights, precision weights or inverse variance weights. 
       `FrequencyWeights` describe the number of times (or frequency) each observation was seen. 
       `ProbabilityWeights` represent the inverse of the sampling probability for each observation,
       providing a correction mechanism for under- or over-sampling certain population groups. `UnitWeights` 
       (default) describes the case in which all weights are equal to 1 (so no weighting takes place)
    - `contrasts::AbstractDict{Symbol}`: a `Dict` mapping term names
      (as `Symbol`s) to term types (e.g., `ContinuousTerm`) or contrasts
      (e.g., `HelmertCoding()`, `SeqDiffCoding(; levels=["a", "b", "c"])`,
      etc.). If contrasts are not provided for a variable, the appropriate
      term type will be guessed based on the data type from the data column:
      any numeric data is assumed to be continuous, and any non-numeric data
      is assumed to be categorical (with `DummyCoding()` as the default contrast type).
    """

include("linpred.jl")
include("lm.jl")
include("glmtools.jl")
include("glmfit.jl")
include("ftest.jl")
include("negbinfit.jl")
include("deprecated.jl")

end # module