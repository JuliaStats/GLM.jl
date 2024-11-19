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
```jldoctest
julia> using RDatasets

julia> df = RDatasets.dataset("mlmRev", "Oxboys");

julia> fit(LinearModel, hcat(ones(nrow(df)), df.Age), df.Height)
LinearModel

Coefficients:
─────────────────────────────────────────────────────────────────
        Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────
x1  149.372      0.528565  282.60    <1e-99  148.33     150.413
x2    6.52102    0.816987    7.98    <1e-13    4.91136    8.13068
─────────────────────────────────────────────────────────────────
```

This model can also be fit as
```jldoctest
julia> using GLM, StableRNGs

julia> lm(hcat(ones(10), 1:10), randn(MersenneTwister(12321), 10))
LinearModel

Coefficients:
────────────────────────────────────────────────────────────────
        Coef.  Std. Error      t  Pr(>|t|)  Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────
x1   0.361896    0.69896    0.52    0.6186  -1.24991    1.9737
x2  -0.012125    0.112648  -0.11    0.9169  -0.271891   0.247641
────────────────────────────────────────────────────────────────
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
GLM.mueta
GLM.inverselink
canonicallink
GLM.glmvar
GLM.mustart
devresid
GLM.dispersion_parameter
GLM.loglik_obs
GLM.cancancel
```
