module GLM
    using Distributions, LinearAlgebra, Printf, Reexport, SparseArrays, Statistics, StatsBase, StatsFuns
    using LinearAlgebra: copytri!, QRCompactWY, Cholesky, CholeskyPivoted, BlasReal
    using Printf: @sprintf
    using StatsBase: CoefTable, StatisticalModel, RegressionModel
    using StatsFuns: logit, logistic
    @reexport using StatsModels
    using Distributions: sqrt2, sqrt2π

    import Base: (\), convert, show, size
    import LinearAlgebra: cholesky, cholesky!
    import Statistics: cor
    import StatsBase: coef, coeftable, coefnames, confint, deviance, nulldeviance, dof, dof_residual,
                      loglikelihood, nullloglikelihood, nobs, stderror, vcov,
                      residuals, predict, predict!,
                      fitted, fit, model_response, response, modelmatrix, r2, r², adjr2, adjr², PValue
    import StatsFuns: xlogy
    import SpecialFunctions: erfc, erfcinv, digamma, trigamma
    import StatsModels: hasintercept
    import Tables
    export coef, coeftable, confint, deviance, nulldeviance, dof, dof_residual,
           loglikelihood, nullloglikelihood, nobs, stderror, vcov, residuals, predict,
           fitted, fit, fit!, model_response, response, modelmatrix, r2, r², adjr2, adjr²,
           cooksdistance, hasintercept, dispersion

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
        linpred,        # linear predictor
        lm,             # linear model
        negbin,         # interface to fitting negative binomial regression
        nobs,           # total number of observations
        predict,        # make predictions
        ftest           # compare models with an F test

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

    @static if VERSION < v"1.8.0-DEV.1139"
        pivoted_cholesky!(A; kwargs...) = cholesky!(A, Val(true); kwargs...)
    else
        pivoted_cholesky!(A; kwargs...) = cholesky!(A, RowMaximum(); kwargs...)
    end

    const COMMON_FIT_KWARGS_DOCS = """
        - `wts::Vector=similar(y,0)`: Prior frequency (a.k.a. case) weights of observations.
          Such weights are equivalent to repeating each observation a number of times equal
          to its weight. Do note that this interpretation gives equal point estimates but
          different standard errors from analytical (a.k.a. inverse variance) weights and
          from probability (a.k.a. sampling) weights which are the default in some other
          software.
          Can be length 0 to indicate no weighting (default).
        - `contrasts::AbstractDict{Symbol}=Dict{Symbol,Any}()`: a `Dict` mapping term names
          (as `Symbol`s) to term types (e.g. `ContinuousTerm`) or contrasts
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
