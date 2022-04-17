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
- `adjr2`:  adjusted R² for a linear model
- `bic`: Bayesian Information Criterion, defined as ``-2 \\log L + k \\log n``, with ``L``
the likelihood of the model, ``k`` is the number of consumed degrees of freedom
- `coef`: extract the estimates of the coefficients in the model
- `confint`: compute confidence intervals for coefficients, with confidence level `level` (by default 95%)
- `cooksdistance`: compute [Cook's distance](https://en.wikipedia.org/wiki/Cook%27s_distance) for each observation in linear model `obj`, giving an estimate of the influence of each data point. Currently only implemented for linear models without weights.
- `deviance`: measure of the model fit, weighted residual sum of squares for lm's
- `dof`: return the number of degrees of freedom consumed in the model, including
when applicable the intercept and the distribution's dispersion parameter
- `dof_residual`: degrees of freedom for residuals, when meaningful
- `fitted`: return the fitted values of the model
- `glm`: fit a generalized linear model (an alias for `fit(GeneralizedLinearModel, ...)`)
- `aic`: Akaike's Information Criterion, defined as ``-2 \\log L + 2k``, with ``L`` the likelihood of the model, and `k` it the number of consumed degrees of freedom
- `aicc`: corrected Akaike's Information Criterion for small sample sizes (Hurvich and Tsai 1989)
- `lm`: fit a linear model (an alias for `fit(LinearModel, ...)`)
- `loglikelihood`: return the log-likelihood of the model
- `modelmatrix`: return the design matrix
- `nobs`: return the number of rows, or sum of the weights when prior weights are specified
- `nulldeviance`: return the deviance of the linear model which includs the intercept only
- `nullloglikelihood`: return the log-likelihood of the null model corresponding to the fitted linear model
- `predict` : obtain predicted values of the dependent variable from the fitted model
- `r2`: R² of a linear model
- `residuals`: get the vector of residuals from the fitted model
- `response`: return the model response (a.k.a the dependent variable)
- `stderror`: standard errors of the coefficients
- `vcov`: estimated variance-covariance matrix of the coefficient estimates


Note that the canonical link for negative binomial regression is `NegativeBinomialLink`, but
in practice one typically uses `LogLink`.

```jldoctest methods
julia> using GLM, DataFrames
julia> data = DataFrame(X=[1,2,3], y=[2,4,7])
julia> test_data = DataFrame(X=[4])
julia> mdl = lm(@formula(y ~  X), data)
julia> r2(mdl)
0.9868421052631579

julia> adjr2(mdl)
0.9736842105263157

julia> bic(mdl)
3.1383527915438716

julia> coef(mdl)
2-element Vector{Float64}:
 -0.6666666666666728
  2.500000000000003

julia> confint(mdl, level=0.90)
2×2 Matrix{Float64}:
 -4.60398   3.27065
  0.677377  4.32262

julia> deviance(mdl)
0.16666666666666666

julia> dof(mdl)
3

julia> dof_residual(mdl)
1.0

julia> aic(mdl)
5.8425159255395425

julia> aicc(mdl)
-18.157484074460456

julia> loglikelihood(mdl)
0.07874203723022877

julia> nullloglikelihood(mdl)
-6.417357973199268
```
`predict` method returns predicted values of response variable from covariate values `newX`.
If you ommit `newX` then it return fitted response values.

```jldoctest methods
julia> predict(mdl)
3-element Vector{Float64}:
 1.8333333333333304
 4.333333333333333
 6.833333333333336

julia> predict(mdl, test_data)
1-element Vector{Union{Missing, Float64}}:
 9.33333333333334

julia> stderror(mdl)
2-element Vector{Float64}:
 0.6236095644623237
 0.2886751345948129
```
`cooksdistance` method computes [Cook's distance](https://en.wikipedia.org/wiki/Cook%27s_distance) for each observation in linear model `obj`, giving an estimate of the influence of each data point. Currently only implemented for linear models without weights.

```jldoctest methods
julia> cooksdistance(mdl)
3-element Vector{Float64}:
 2.500000000000079
 0.2499999999999991
 2.499999999999919
```

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
