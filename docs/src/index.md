# GLM.jl Manual

Linear and generalized linear models in Julia

## Installation

```julia
Pkg.add("GLM")
```

will install this package and its dependencies, which includes the [Distributions package](https://github.com/JuliaStats/Distributions.jl).

The [RDatasets package](https://github.com/JuliaStats/RDatasets.jl) is useful for fitting models on standard R datasets to compare the results with those from R.

## Fitting models

Two methods taking different kinds of arguments can be used to fit a model:
- for linear models: `lm(formula, data)` and `lm(X, y)`;
- for generalized linear models (GLM): `glm(formula, data, family, link)` and `glm(X, y, family, link)`.

Their arguments must be:

- `formula`: a [StatsModels.jl `Formula` object](https://juliastats.org/StatsModels.jl/stable/formula/)
  referring to columns in `data`; for example, if column names are `:Y`, `:X1`, and `:X2`,
  then a valid formula is `@formula(Y ~ X1 + X2)`
- `data`: a table in the Tables.jl definition, e.g. a data frame;
  rows with `missing` values are ignored
- `X` a matrix holding values of the independent variable(s) in columns
- `y` a vector holding values of the dependent variable
  (including if appropriate the intercept)
- `family`: chosen from `Bernoulli()`, `Binomial()`, `Gamma()`, `Geometric()`, `Normal()`, `Poisson()`, or `NegativeBinomial(θ)`
- `link`: chosen from the list below, for example, `LogitLink()` is a valid link for the `Binomial()` family

Typical distributions for use with `glm` and their canonical link
functions are

           Bernoulli (LogitLink)
            Binomial (LogitLink)
               Gamma (InverseLink)
           Geometric (LogLink)
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
    PowerLink
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
LinearModel

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
LinearModel

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

## Weighting

Both `lm` and `glm` allow weighted estimation. The four different
[types of weights](https://juliastats.org/StatsBase.jl/stable/weights/) defined in
[StatsBase.jl](https://github.com/JuliaStats/StatsBase.jl) can be used to fit a model:

- `AnalyticWeights` describe a non-random relative importance (usually between 0 and 1) for
  each observation. These weights may also be referred to as reliability weights, precision
  weights or inverse variance weights. These are typically used when the observations being
  weighted are aggregate values (e.g., averages) with differing variances.
- `FrequencyWeights` describe the number of times (or frequency) each observation was seen.
  These weights may also be referred to as case weights or repeat weights.
- `ProbabilityWeights` represent the inverse of the sampling probability for each observation,
  providing a correction mechanism for under- or over-sampling certain population groups.
  These weights may also be referred to as sampling weights.
- `UnitWeights` attribute a weight of 1 to each observation, which corresponds
  to unweighted regression (the default).

To indicate which kind of weights should be used, the vector of weights must be wrapped in
one of the three weights types, and then passed to the `weights` keyword argument.
Short-hand functions `aweights`, `fweights`, and `pweights` can be used to construct
`AnalyticWeights`, `FrequencyWeights`, and `ProbabilityWeights`, respectively.

Using analytic weights corresponds to weighted least squares.
This gives the same results as R and Stata.

Probability weights give the same point estimates as analytic weights, but standard errors
and p-values are based on a sandwich (heteroskedasticity-robust) estimator.
This gives the same results as the R `survey` package with a simple survey design
without strata nor clustering, but differs from Stata with the `pweights` option, which
adopts the same approach but with a different assumption regarding degrees of freedom.

We illustrate the API with randomly generated data.

```jldoctest weights
julia> using StableRNGs, DataFrames, StatsBase, GLM

julia> data = DataFrame(y = rand(StableRNG(1), 100), x = randn(StableRNG(2), 100), weights = repeat([1, 2, 3, 4], 25));

julia> m = lm(@formula(y ~ x), data)
LinearModel

y ~ 1 + x

Coefficients:
──────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      t  Pr(>|t|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)   0.517369    0.0280232  18.46    <1e-32   0.461758  0.57298
x            -0.0500249   0.0307201  -1.63    0.1066  -0.110988  0.0109382
──────────────────────────────────────────────────────────────────────────

julia> m_aweights = lm(@formula(y ~ x), data, wts=aweights(data.weights))
LinearModel

y ~ 1 + x

Coefficients:
──────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      t  Pr(>|t|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)   0.51673     0.0270707  19.09    <1e-34   0.463009  0.570451
x            -0.0478667   0.0308395  -1.55    0.1239  -0.109067  0.0133333
──────────────────────────────────────────────────────────────────────────

julia> m_fweights = lm(@formula(y ~ x), data, wts=fweights(data.weights))
LinearModel

y ~ 1 + x

Coefficients:
─────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      t  Pr(>|t|)   Lower 95%    Upper 95%
─────────────────────────────────────────────────────────────────────────────
(Intercept)   0.51673     0.0170172  30.37    <1e-84   0.483213    0.550246
x            -0.0478667   0.0193863  -2.47    0.0142  -0.0860494  -0.00968394
─────────────────────────────────────────────────────────────────────────────

julia> m_pweights = lm(@formula(y ~ x), data, wts=pweights(data.weights))
LinearModel

y ~ 1 + x

Coefficients:
───────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      t  Pr(>|t|)  Lower 95%   Upper 95%
───────────────────────────────────────────────────────────────────────────
(Intercept)   0.51673     0.0287193  17.99    <1e-32   0.459737  0.573722
x            -0.0478667   0.0265532  -1.80    0.0745  -0.100561  0.00482739
───────────────────────────────────────────────────────────────────────────

```

!!! note
    In the old API before GLM.jl 2.0, weights were passed as `AbstractVectors`
    and were silently treated in the internal computation of standard errors
    and related quantities as `FrequencyWeights`.

The type of the weights will affect the variance of the estimated coefficients and the
quantities involving this variance. The coefficient point estimates will be the same
regardless of the type of weights.

```jldoctest weights; filter = r"(\d*)\.(\d{10})\d+" => s"\1.\2***"
julia> loglikelihood(m_aweights)
-16.296307561384253

julia> loglikelihood(m_fweights)
-25.518609617564483
```

## Comparing models with F-test

Comparisons between two or more linear models can be performed using the `ftest` function,
which computes an F-test between each pair of subsequent models and reports fit statistics:

```jldoctest
julia> using DataFrames, GLM, StableRNGs

julia> data = DataFrame(y = (1:50).^2 .+ randn(StableRNG(1), 50), x = 1:50);

julia> ols_lin = lm(@formula(y ~ x), data);

julia> ols_sq = lm(@formula(y ~ x + x^2), data);

julia> ftest(ols_lin, ols_sq)
F-test: 2 models fitted on 50 observations
────────────────────────────────────────────────────────────────────────────────
     DOF  ΔDOF           SSR          ΔSSR      R²     ΔR²            F*   p(>F)
────────────────────────────────────────────────────────────────────────────────
[1]    3        1731979.2266                0.9399
[2]    4     1       40.7581  1731938.4685  1.0000  0.0601  1997177.0357  <1e-99
────────────────────────────────────────────────────────────────────────────────
```

## Methods applied to fitted models

Many of the methods provided by this package have names similar to those in [R](http://www.r-project.org).

- `adjr2`: adjusted R² for a linear model (an alias for `adjr²`)
- `aic`: Akaike's Information Criterion
- `aicc`: corrected Akaike's Information Criterion for small sample sizes
- `bic`: Bayesian Information Criterion
- `coef`: estimates of the coefficients in the model
- `confint`: confidence intervals for coefficients
- `cooksdistance`: [Cook's distance](https://en.wikipedia.org/wiki/Cook%27s_distance) for each observation
- `deviance`: measure of the model fit, weighted residual sum of squares for lm's
- `dispersion`: dispersion (or scale) parameter for a model's distribution
- `dof`: number of degrees of freedom consumed in the model
- `dof_residual`: degrees of freedom for residuals, when meaningful
- `fitted`: fitted values of the model
- `glm`: fit a generalized linear model (an alias for `fit(GeneralizedLinearModel, ...)`)
- `lm`: fit a linear model (an alias for `fit(LinearModel, ...)`)
- `loglikelihood`: log-likelihood of the model
- `modelmatrix`: design matrix
- `nobs`: number of rows, or sum of the weights when prior weights are specified
- `nulldeviance`: deviance of the model with all predictors removed
- `nullloglikelihood`: log-likelihood of the model with all predictors removed
- `predict`: predicted values of the dependent variable from the fitted model
- `r2`: R² of a linear model (an alias for `r²`)
- `residuals`: vector of residuals from the fitted model
- `response`: model response (a.k.a the dependent variable)
- `stderror`: standard errors of the coefficients
- `vcov`: variance-covariance matrix of the coefficient estimates

```jldoctest methods
julia> using GLM, DataFrames, StatsBase

julia> data = DataFrame(X=[1,2,3], y=[2,4,7]);

julia> mdl = lm(@formula(y ~ X), data);

julia> round.(coef(mdl); digits=8)
2-element Vector{Float64}:
 -0.66666667
  2.5

julia> round(r2(mdl); digits=8)
0.98684211

julia> round(aic(mdl); digits=8)
5.84251593
```

The [`predict`](@ref) method returns predicted values of response variable from covariate values in an input `newX`.
If `newX` is omitted then the fitted response values from the model are returned.

```jldoctest methods
julia> test_data = DataFrame(X=[4]);

julia> round.(predict(mdl, test_data); digits=8)
1-element Vector{Float64}:
 9.33333333
```

The [`cooksdistance`](@ref) method computes
[Cook's distance](https://en.wikipedia.org/wiki/Cook%27s_distance) for each observation
used to fit a linear model, giving an estimate of the influence of each data point.
Note that it's currently only implemented for linear models without weights.

```jldoctest methods
julia> round.(cooksdistance(mdl); digits=8)
3-element Vector{Float64}:
 2.5
 0.25
 2.5
```

## Debugging failed fits

In the rare cases when a fit of a generalized linear model fails, it can be useful
to enable more output from the fitting steps. This can be done through
the Julia logging mechanism by setting `ENV["JULIA_DEBUG"] = GLM`. Enabling debug output
will result in ouput like the following

```julia
┌ Debug: Iteration: 1, deviance: 5.129147109764238, diff.dev.:0.05057195315968688
└ @ GLM ~/.julia/dev/GLM/src/glmfit.jl:418
┌ Debug: Iteration: 2, deviance: 5.129141077001254, diff.dev.:6.032762984276019e-6
└ @ GLM ~/.julia/dev/GLM/src/glmfit.jl:418
┌ Debug: Iteration: 3, deviance: 5.129141077001143, diff.dev.:1.1102230246251565e-13
└ @ GLM ~/.julia/dev/GLM/src/glmfit.jl:418
```
