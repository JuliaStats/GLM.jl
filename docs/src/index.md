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
    NegativeBinomial (NegativeBinomialLink, often used with LogLink)
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

Note that the canonical link for negative binomial regression is `NegativeBinomialLink`, but
in practice one typically uses `LogLink`.
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
julia> using CategoricalArrays, DataFrames, GLM, StableRNGs

julia> rng = StableRNG(1); # Ensure example can be reproduced

julia> data = DataFrame(y = rand(rng, 100), x = categorical(repeat([1, 2, 3, 4], 25)));


julia> lm(@formula(y ~ x), data)
StatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}}}}, Matrix{Float64}}

y ~ 1 + x

Coefficients:
───────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      t  Pr(>|t|)   Lower 95%  Upper 95%
───────────────────────────────────────────────────────────────────────────
(Intercept)   0.490985    0.0564176   8.70    <1e-13   0.378997    0.602973
x: 2          0.0527655   0.0797865   0.66    0.5100  -0.105609    0.21114
x: 3          0.0955446   0.0797865   1.20    0.2341  -0.0628303   0.25392
x: 4         -0.032673    0.0797865  -0.41    0.6831  -0.191048    0.125702
───────────────────────────────────────────────────────────────────────────
```

Using [`contrasts`](https://juliastats.github.io/StatsModels.jl/stable/contrasts/):

```jldoctest categorical
julia> using StableRNGs

julia> data = DataFrame(y = rand(StableRNG(1), 100), x = repeat([1, 2, 3, 4], 25));

julia> lm(@formula(y ~ x), data, contrasts = Dict(:x => DummyCoding()))
StatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}}}}, Matrix{Float64}}

y ~ 1 + x

Coefficients:
───────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      t  Pr(>|t|)   Lower 95%  Upper 95%
───────────────────────────────────────────────────────────────────────────
(Intercept)   0.490985    0.0564176   8.70    <1e-13   0.378997    0.602973
x: 2          0.0527655   0.0797865   0.66    0.5100  -0.105609    0.21114
x: 3          0.0955446   0.0797865   1.20    0.2341  -0.0628303   0.25392
x: 4         -0.032673    0.0797865  -0.41    0.6831  -0.191048    0.125702
───────────────────────────────────────────────────────────────────────────
```

## Comparing models with F-test

Comparisons between two or more linear models can be performed using the `ftest` function,
which computes an F-test between each pair of subsequent models and reports fit statistics:
```jldoctest
julia> using DataFrames, GLM, StableRNGs

julia> data = DataFrame(y = (1:50).^2 .+ randn(StableRNG(1), 50), x = 1:50);

julia> ols_lin = lm(@formula(y ~ x), data);

julia> ols_sq = lm(@formula(y ~ x + x^2), data);

julia> ftest(ols_lin.model, ols_sq.model)
F-test: 2 models fitted on 50 observations
─────────────────────────────────────────────────────────────────────────────────
     DOF  ΔDOF           SSR           ΔSSR      R²     ΔR²            F*   p(>F)
─────────────────────────────────────────────────────────────────────────────────
[1]    3        1731979.2266                 0.9399
[2]    4     1       40.7581  -1731938.4685  1.0000  0.0601  1997177.0357  <1e-99
─────────────────────────────────────────────────────────────────────────────────
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
- `residuals`: get the vector of residuals from the fitted model

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
