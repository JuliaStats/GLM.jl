# API

```@meta
DocTestSetup = quote
    using CategoricalArrays, DataFrames, Distributions, GLM, RDatasets
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
julia> using Random

julia> fit(LinearModel, hcat(ones(10), 1:10), randn(MersenneTwister(12321), 10))
LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}:

Coefficients:
────────────────────────────────────────────────────────────────
        Coef.  Std. Error      t  Pr(>|t|)  Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────
x1  -0.680621    0.702626  -0.97    0.3611  -2.30088    0.939637
x2   0.053912    0.113238   0.48    0.6467  -0.207216   0.31504
────────────────────────────────────────────────────────────────
```

This model can also be fit as
```jldoctest
julia> using Random

julia> lm(hcat(ones(10), 1:10), randn(MersenneTwister(12321), 10))
LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}:

Coefficients:
────────────────────────────────────────────────────────────────
        Coef.  Std. Error      t  Pr(>|t|)  Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────
x1  -0.680621    0.702626  -0.97    0.3611  -2.30088    0.939637
x2   0.053912    0.113238   0.48    0.6467  -0.207216   0.31504
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
StatsBase.deviance
GLM.dispersion
GLM.ftest
GLM.installbeta!
StatsBase.nobs
StatsBase.nulldeviance
StatsBase.predict
StatsModels.isnested
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
