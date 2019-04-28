# API

```@meta
DocTestSetup = quote
    using CategoricalArrays, DataFrames, Distributions, GLM, RDatasets
end
```

## Types defined in the package

```@docs
DensePredChol
DensePredQR
GlmResp
LinearModel
LmResp
LinPred
GLM.ModResp
```

## Constructors for models

The most general approach to fitting a model is with the `fit` function, as in
```jldoctest
julia> using Random

julia> fit(LinearModel, hcat(ones(10), 1:10), randn(MersenneTwister(12321), 10))
LinearModel{GLM.LmResp{Array{Float64,1}},GLM.DensePredChol{Float64,LinearAlgebra.Cholesky{Float64,Array{Float64,2}}}}:

Coefficients:
────────────────────────────────────────────────────────────────
        Coef.  Std. Error      t  Pr(>|t|)  Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────
x1   0.717436    0.775175   0.93    0.3818  -1.07012    2.50499
x2  -0.152062    0.124931  -1.22    0.2582  -0.440153   0.136029
────────────────────────────────────────────────────────────────
```

This model can also be fit as
```jldoctest
julia> using Random

julia> lm(hcat(ones(10), 1:10), randn(MersenneTwister(12321), 10))
LinearModel{LmResp{Array{Float64,1}},DensePredChol{Float64,LinearAlgebra.Cholesky{Float64,Array{Float64,2}}}}:

Coefficients:
────────────────────────────────────────────────────────────────
        Coef.  Std. Error      t  Pr(>|t|)  Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────
x1   0.717436    0.775175   0.93    0.3818  -1.07012    2.50499
x2  -0.152062    0.124931  -1.22    0.2582  -0.440153   0.136029
────────────────────────────────────────────────────────────────
```

```@docs
glm
fit
lm
```

## Model methods
```@docs
GLM.cancancel
delbeta!
StatsBase.deviance
GLM.dispersion
GLM.installbeta!
GLM.issubmodel
linpred!
linpred
StatsBase.nobs
StatsBase.nulldeviance
StatsBase.predict
updateμ!
wrkresp
GLM.wrkresp!
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
ProbitLink
SqrtLink
linkfun
linkinv
mueta
inverselink
canonicallink
glmvar
mustart
devresid
GLM.dispersion_parameter
GLM.loglik_obs
```
