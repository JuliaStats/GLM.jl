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
    using StatsAPI
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
        linpred,        # linear predictor
        lm,             # linear model
        negbin,         # interface to fitting negative binomial regression
        nobs,           # total number of observations
        predict,        # make predictions
        ftest           # compare models with an F test

    # Plot functions
    export cooksleverageplot, cooksleverageplot!
    export scalelocationplot, scalelocationplot!
    export residualplot, residualplot!
    export residualsleverageplot, residualsleverageplot!
    export lmplot


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

    @static if VERSION < v"1.7.0"
        pivoted_qr!(A; kwargs...) = qr!(A, Val(true); kwargs...)
    else
        pivoted_qr!(A; kwargs...) = qr!(A, ColumnNorm(); kwargs...)
    end

    if !isdefined(Base, :get_extension)
        using Requires
    end

    function __init__()
        @static if !isdefined(Base, :get_extension)
            @require StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd" include("../ext/StatsPlotsExt.jl")
            @require Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" include("../ext/MakieExt.jl")
        end
    end

    const COMMON_FIT_KWARGS_DOCS = """
        - `dropcollinear::Bool=true`: Controls whether or not a model matrix
          less-than-full rank is accepted.
          If `true` (the default) the coefficient for redundant linearly dependent columns is
          `0.0` and all associated statistics are set to `NaN`.
          Typically from a set of linearly-dependent columns the last ones are identified as redundant
          (however, the exact selection of columns identified as redundant is not guaranteed).
        - `method::Symbol=:cholesky`: Controls which decomposition method to use.
          If `method=:cholesky` (the default), then the `Cholesky` decomposition method will be used.
          If `method=:qr`, then the `QR` decomposition method (which is more stable
          but slower) will be used.
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
    include("plots.jl")

end # module
