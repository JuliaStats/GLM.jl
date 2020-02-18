# Manual

## Installation

```julia
Pkg.add("GLM")
```

will install this package and its dependencies, which includes the [Distributions package](https://github.com/JuliaStats/Distributions.jl).

The [RDatasets package](https://github.com/johnmyleswhite/RDatasets.jl) is useful for fitting models on standard R datasets to compare the results with those from R.

## Fitting GLM models

To fit a Generalized Linear Model (GLM), use the function, `glm(formula, data, family, link)`, where,
- `formula`: uses column symbols from the DataFrame data, for example, if `names(data)=[:Y,:X1,:X2]`, then a valid formula is `@formula(Y ~ X1 + X2)`
- `data`: a DataFrame which may contain NA values, any rows with NA values are ignored
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
[Tables.jl](https://juliadata.github.io/Tables.jl/stable/) table (DataFrames, JuliaDB,
ColumnTable, etc). Alternatively, you can pass an explicit 
[contrasts](https://juliastats.github.io/StatsModels.jl/stable/contrasts/) argument if you
would like a different contrast coding system or if you are not using DataFrames.

The response (dependent) variable may not be categorical.

Using a `CategoricalVector` constructed with `categorical` or `categorical!`:

```jldoctest
julia> using DataFrames, GLM

julia> data = DataFrame(y = rand(100), x = categorical(repeat([1, 2, 3, 4], 25)));

julia> lm(@formula(y ~ x), data)
StatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Array{Float64,1}},GLM.DensePredChol{Float64, LinearAlgebra.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}

y ~ 1 + x

Coefficients:
──────────────────────────────────────────────────────────────────────────────
               Estimate  Std. Error    t value  Pr(>|t|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────────
(Intercept)   0.561258    0.0612945   9.15675     <1e-14   0.439589  0.682927
x: 2         -0.121297    0.0866835  -1.39931     0.1649  -0.293363  0.0507681
x: 3         -0.109711    0.0866835  -1.26565     0.2087  -0.281776  0.0623548
x: 4          0.0490837   0.0866835   0.566241    0.5726  -0.122982  0.221149
──────────────────────────────────────────────────────────────────────────────
```

Using [`contrasts`](https://juliastats.github.io/StatsModels.jl/stable/contrasts/):

```jldoctest
julia> using DataFrames, GLM

julia> data = DataFrame(y = rand(100), x = repeat([1, 2, 3, 4], 25));

julia> lm(@formula(y ~ x), data, contrasts = Dict(:x => DummyCoding()))
StatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Array{Float64,1}},GLM.DensePredChol{Float64, LinearAlgebra.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}

y ~ 1 + x

Coefficients:
────────────────────────────────────────────────────────────────────────────────
              Estimate  Std. Error    t value  Pr(>|t|)     Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────────────────────
(Intercept)  0.404169    0.0545628  7.40741      <1e-10   0.295863      0.512476
x: 2         0.0048212   0.0771635  0.0624804    0.9503  -0.148347      0.15799
x: 3         0.153049    0.0771635  1.98344      0.0502  -0.000118963   0.306218
x: 4         0.0512872   0.0771635  0.664656     0.5079  -0.101881      0.204456
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
