# GLM.jl Documentation

```@meta
DocTestSetup = quote
    using Distributions, GLM
end
```

## Types defined in the package

```@docs
LinearModel
LmResp
LinPred
GlmResp
DensePredQR
DensePredChol
```

## Constructors for models

The most general approach to fitting a model is with the [`StatsBase.fit`](@ref) function, as in
```jldoctest
julia> fit(LinearModel, hcat(ones(10), 1:10), randn(MersenneTwister(12321), 10))
GLM.LinearModel{GLM.LmResp{Array{Float64,1}},GLM.DensePredChol{Float64,Base.LinAlg.Cholesky{Float64,Array{Float64,2}}}}:

Coefficients:
      Estimate Std.Error  t value Pr(>|t|)
x1    0.717436  0.775175 0.925515   0.3818
x2   -0.152062  0.124931 -1.21717   0.2582
```

This model can also be fit as
```jldoctest
julia> lm(hcat(ones(10), 1:10), randn(MersenneTwister(12321), 10))
GLM.LinearModel{GLM.LmResp{Array{Float64,1}},GLM.DensePredChol{Float64,Base.LinAlg.Cholesky{Float64,Array{Float64,2}}}}:

Coefficients:
      Estimate Std.Error  t value Pr(>|t|)
x1    0.717436  0.775175 0.925515   0.3818
x2   -0.152062  0.124931 -1.21717   0.2582
```

## Methods for model updating
```@docs
delbeta!
linpred!
linpred
GLM.installbeta!
GLM.cancancel
updateμ!
wrkresp
GLM.wrkresp!
GLM.dispersion
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
