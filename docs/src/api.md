# API

```@meta
DocTestSetup = quote
    using CategoricalArrays, DataFrames, Distributions, GLM, RDatasets, StableRNGs
end
```

## Types defined in the package

```@docs
LinearModel
GLM.DensePredChol
GLM.DensePredQR
GLM.LmResp
GLM.GlmResp
GLM.LinPred
GLM.ModResp
```

## Constructors for models

The most general approach to fitting a model is with the `fit` function, as in
```jldoctest constructors
julia> using RDatasets

julia> df = RDatasets.dataset("mlmRev", "Oxboys");

julia> fit(LinearModel, @formula(Height ~ Age), df)
LinearModel

Formula: Height ~ 1 + Age

Coefficients:
(Intercept)    Age
      149.4  6.521

Number of observations:                     234
Residual degrees of freedom:                232
Residual deviance:                      15148.5
```

This model can also be fit as
```jldoctest constructors
julia> lm(@formula(Height ~ Age), df)
LinearModel

Formula: Height ~ 1 + Age

Coefficients:
(Intercept)    Age
      149.4  6.521

Number of observations:                     234
Residual degrees of freedom:                232
Residual deviance:                      15148.5
```

```@docs
lm
glm
negbin
fit
```

## Model methods
```@docs
cooksdistance
StatsBase.deviance
GLM.dispersion
GLM.ftest
StatsBase.nobs
StatsBase.nulldeviance
StatsBase.predict
```

## Links and methods applied to them
```@docs
Link
GLM.Link01
CauchitLink
CloglogLink
IdentityLink
InverseLink
InverseSquareLink
LogitLink
LogLink
NegativeBinomialLink
PowerLink
ProbitLink
SqrtLink
GLM.linkfun
GLM.linkinv
GLM.inverselink
canonicallink
GLM.glmvar
GLM.mustart
devresid
GLM.dispersion_parameter
GLM.loglik_obs
GLM.cancancel
```
