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
