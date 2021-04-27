# GLM.jl Manual

Linear and generalized linear models in Julia

## Installation

```julia
Pkg.add("GLM")
```

will install this package and its dependencies, which includes the [Distributions package](https://github.com/JuliaStats/Distributions.jl).

The [RDatasets package](https://github.com/johnmyleswhite/RDatasets.jl) is useful for fitting models on standard R datasets to compare the results with those from R.

## Fitting GLM models

Two methods can be used to fit a Generalized Linear Model (GLM):
`glm(formula, data, family, link)` and `glm(X, y, family, link)`.
Their arguments must be:
- `formula`: a [StatsModels.jl `Formula` object](https://juliastats.org/StatsModels.jl/stable/formula/)
  referring to columns in `data`; for example, if column names are `:Y`, `:X1`, and `:X2`,
  then a valid formula is `@formula(Y ~ X1 + X2)`
- `data`: a table in the Tables.jl definition, e.g. a data frame;
  rows with `missing` values are ignored
- `X` a matrix holding values of the dependent variable(s) in columns
- `y` a vector holding values of the independent variable
  (including if appropriate the intercept)
- `family`: chosen from `Bernoulli()`, `Binomial()`, `Gamma()`, `Normal()`, `Poisson()`, or `NegativeBinomial(θ)`
- `link`: chosen from the list below, for example, `LogitLink()` is a valid link for the `Binomial()` family

Typical distributions for use with `glm` and their canonical link
functions are

           Bernoulli (LogitLink)
            Binomial (LogitLink)
               Gamma (InverseLink)
     InverseGaussian (InverseSquareLink)
    NegativeBinomial (LogLink)
              Normal (IdentityLink)
             Poisson (LogLink)

Currently the available Link types are

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

The `NegativeBinomial` distribution belongs to the exponential family only if θ (the shape
parameter) is fixed, thus θ has to be provided if we use `glm` with `NegativeBinomial` family.
If one would like to also estimate θ, then `negbin(formula, data, link)` should be
used instead.

An intercept is included in any GLM by default.

## Categorical variables

Categorical variables will be dummy coded by default if they are non-numeric or if they are
[`CategoricalVector`s](https://juliadata.github.io/CategoricalArrays.jl/stable/) within a
[Tables.jl](https://juliadata.github.io/Tables.jl/stable/) table (`DataFrame`, JuliaDB table,
named tuple of vectors, etc). Alternatively, you can pass an explicit 
[contrasts](https://juliastats.github.io/StatsModels.jl/stable/contrasts/) argument if you
would like a different contrast coding system or if you are not using DataFrames.

The response (dependent) variable may not be categorical.

Using a `CategoricalVector` constructed with `categorical` or `categorical!`:

```jldoctest categorical
julia> using DataFrames, GLM, Random

julia> Random.seed!(1); # Ensure example can be reproduced

julia> data = DataFrame(y = rand(100), x = categorical(repeat([1, 2, 3, 4], 25)));

julia> lm(@formula(y ~ x), data)
StatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Array{Float64,1}},GLM.DensePredChol{Float64,LinearAlgebra.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}

y ~ 1 + x

Coefficients:
─────────────────────────────────────────────────────────────────────────────
              Estimate  Std. Error   t value  Pr(>|t|)   Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────────
(Intercept)  0.41335     0.0548456  7.53662     <1e-10   0.304483    0.522218
x: 2         0.172338    0.0775634  2.2219      0.0286   0.0183756   0.3263  
x: 3         0.0422104   0.0775634  0.544205    0.5876  -0.111752    0.196172
x: 4         0.0793591   0.0775634  1.02315     0.3088  -0.074603    0.233321
─────────────────────────────────────────────────────────────────────────────
```

Using [`contrasts`](https://juliastats.github.io/StatsModels.jl/stable/contrasts/):

```jldoctest categorical
julia> data = DataFrame(y = rand(100), x = repeat([1, 2, 3, 4], 25));

julia> lm(@formula(y ~ x), data, contrasts = Dict(:x => DummyCoding()))
StatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Array{Float64,1}},GLM.DensePredChol{Float64,LinearAlgebra.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}

y ~ 1 + x

Coefficients:
────────────────────────────────────────────────────────────────────────────────
               Estimate  Std. Error     t value  Pr(>|t|)   Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────────────────────
(Intercept)   0.464446    0.0582412   7.97453      <1e-11   0.348838    0.580054
x: 2         -0.0057872   0.0823655  -0.0702624    0.9441  -0.169281    0.157707
x: 3          0.0923976   0.0823655   1.1218       0.2647  -0.0710966   0.255892
x: 4          0.115145    0.0823655   1.39797      0.1653  -0.0483494   0.278639
────────────────────────────────────────────────────────────────────────────────
```

## Methods applied to fitted models

Many of the methods provided by this package have names similar to those in [R](http://www.r-project.org).
- `coef`: extract the estimates of the coefficients in the model
- `deviance`: measure of the model fit, weighted residual sum of squares for lm's
- `dof_residual`: degrees of freedom for residuals, when meaningful
- `glm`: fit a generalized linear model (an alias for `fit(GeneralizedLinearModel, ...)`)
- `lm`: fit a linear model (an alias for `fit(LinearModel, ...)`)
- `r2`: R² of a linear model or pseudo-R² of a generalized linear model
- `stderror`: standard errors of the coefficients
- `vcov`: estimated variance-covariance matrix of the coefficient estimates
- `predict` : obtain predicted values of the dependent variable from the fitted model

Note that the canonical link for negative binomial regression is `NegativeBinomialLink`, but
in practice one typically uses `LogLink`.

## Separation of response object and predictor object

The general approach in this code is to separate functionality related
to the response from that related to the linear predictor.  This
allows for greater generality by mixing and matching different
subtypes of the abstract type ```LinPred``` and the abstract type ```ModResp```.

A ```LinPred``` type incorporates the parameter vector and the model
matrix.  The parameter vector is a dense numeric vector but the model
matrix can be dense or sparse.  A ```LinPred``` type must incorporate
some form of a decomposition of the weighted model matrix that allows
for the solution of a system ```X'W * X * delta=X'wres``` where ```W``` is a
diagonal matrix of "X weights", provided as a vector of the square
roots of the diagonal elements, and ```wres``` is a weighted residual vector.

Currently there are two dense predictor types, ```DensePredQR``` and
```DensePredChol```, and the usual caveats apply.  The Cholesky
version is faster but somewhat less accurate than that QR version.
The skeleton of a distributed predictor type is in the code
but not yet fully fleshed out.  Because Julia by default uses
OpenBLAS, which is already multi-threaded on multicore machines, there
may not be much advantage in using distributed predictor types.

A ```ModResp``` type must provide methods for the ```wtres``` and
```sqrtxwts``` generics.  Their values are the arguments to the
```updatebeta``` methods of the ```LinPred``` types.  The
```Float64``` value returned by ```updatedelta``` is the value of the
convergence criterion.

Similarly, ```LinPred``` types must provide a method for the
```linpred``` generic.  In general ```linpred``` takes an instance of
a ```LinPred``` type and a step factor.  Methods that take only an instance
of a ```LinPred``` type use a default step factor of 1.  The value of
```linpred``` is the argument to the ```updatemu``` method for
```ModResp``` types.  The ```updatemu``` method returns the updated
deviance.
