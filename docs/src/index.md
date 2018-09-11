# GLM Documentation

```@meta
DocTestSetup = quote
    if Pkg.installed("RDatasets") isa Void
        Pkg.add("RDatasets")
    end
    using CategoricalArrays, DataFrames, Distributions, GLM, RDatasets
end
```

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

The `NegativeBinomial` distribution belongs to the exponential family only if θ (the shape
parameter) is fixed, thus θ has to be provided if we use `glm` with `NegativeBinomial` family. 
If one would like to also estimate θ, then `negbin(formula, data, link)` should be
used instead.

An intercept is included in any GLM by default.

## Methods applied to fitted models

Many of the methods provided by this package have names similar to those in [R](http://www.r-project.org).
- `coef`: extract the estimates of the coefficients in the model
- `deviance`: measure of the model fit, weighted residual sum of squares for lm's
- `dof_residual`: degrees of freedom for residuals, when meaningful
- `glm`: fit a generalized linear model (an alias for `fit(GeneralizedLinearModel, ...)`)
- `lm`: fit a linear model (an alias for `fit(LinearModel, ...)`)
- `stderror`: standard errors of the coefficients
- `vcov`: estimated variance-covariance matrix of the coefficient estimates
- `predict` : obtain predicted values of the dependent variable from the fitted model

## Minimal examples

### Ordinary Least Squares Regression:
```jldoctest
julia> using DataFrames, GLM

julia> data = DataFrame(X=[1,2,3], Y=[2,4,7])
3×2 DataFrames.DataFrame
│ Row │ X │ Y │
├─────┼───┼───┤
│ 1   │ 1 │ 2 │
│ 2   │ 2 │ 4 │
│ 3   │ 3 │ 7 │

julia> ols = lm(@formula(Y ~ X), data)
StatsModels.DataFrameRegressionModel{GLM.LinearModel{GLM.LmResp{Array{Float64,1}},GLM.DensePredChol{Float64,Base.LinAlg.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}

Formula: Y ~ 1 + X

Coefficients:
              Estimate Std.Error  t value Pr(>|t|)
(Intercept)  -0.666667   0.62361 -1.06904   0.4788
X                  2.5  0.288675  8.66025   0.0732

julia> stderror(ols)
2-element Array{Float64,1}:
 0.62361
 0.288675

julia> predict(ols)
3-element Array{Float64,1}:
 1.83333
 4.33333
 6.83333

julia> newX = DataFrame(X=[2,3,4]);

julia> predict(ols, newX, interval=:confint)
 3×3 Array{Float64,2}:
  4.33333  1.33845   7.32821
  6.83333  2.09801  11.5687
  9.33333  1.40962  17.257
```

The columns of the matrix are prediction, 95% lower and upper confidence bounds
.

### Probit Regression:
```jldoctest
julia> data = DataFrame(X=[1,2,3], Y=[1,0,1])
3×2 DataFrames.DataFrame
│ Row │ X │ Y │
├─────┼───┼───┤
│ 1   │ 1 │ 1 │
│ 2   │ 2 │ 0 │
│ 3   │ 3 │ 1 │

julia> probit = glm(@formula(Y ~ X), data, Binomial(), ProbitLink())
StatsModels.DataFrameRegressionModel{GLM.GeneralizedLinearModel{GLM.GlmResp{Array{Float64,1},Distributions.Binomial{Float64},GLM.ProbitLink},GLM.DensePredChol{Float64,Base.LinAlg.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}

Formula: Y ~ 1 + X

Coefficients:
                 Estimate Std.Error      z value Pr(>|z|)
(Intercept)      0.430727   1.98019     0.217518   0.8278
X            -3.64399e-19   0.91665 -3.97534e-19   1.0000

```

### Negative Binomial Regression:
```jldoctest
julia> using GLM, RDatasets

julia> quine = dataset("MASS", "quine")

julia> nbrmodel = glm(@formula(Days ~ Eth+Sex+Age+Lrn), quine, NegativeBinomial(2.0), LogLink())
StatsModels.DataFrameRegressionModel{GLM.GeneralizedLinearModel{GLM.GlmResp{Array{Float64,1},Distributions.NegativeBinomial{Float64},GLM.LogLink},GLM.DensePredChol{Float64,Base.LinAlg.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}

Formula: Days ~ 1 + Eth + Sex + Age + Lrn

Coefficients:
              Estimate Std.Error  z value Pr(>|z|)
(Intercept)    2.88645  0.227144  12.7076   <1e-36
Eth: N       -0.567515  0.152449 -3.72265   0.0002
Sex: M       0.0870771  0.159025 0.547568   0.5840
Age: F1      -0.445076  0.239087 -1.86157   0.0627
Age: F2      0.0927999  0.234502 0.395731   0.6923
Age: F3       0.359485  0.246586  1.45785   0.1449
Lrn: SL       0.296768  0.185934  1.59609   0.1105

julia> nbrmodel = negbin(@formula(Days ~ Eth+Sex+Age+Lrn), quine, LogLink())
StatsModels.DataFrameRegressionModel{GLM.GeneralizedLinearModel{GLM.GlmResp{Array{Float64,1},Distributions.NegativeBinomial{Float64},GLM.LogLink},GLM.DensePredChol{Float64,Base.LinAlg.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}

Formula: Days ~ 1 + Eth + Sex + Age + Lrn

Coefficients:
              Estimate Std.Error  z value Pr(>|z|)
(Intercept)    2.89453  0.227415   12.728   <1e-36
Eth: N       -0.569341  0.152656 -3.72957   0.0002
Sex: M       0.0823881  0.159209 0.517485   0.6048
Age: F1      -0.448464  0.238687 -1.87888   0.0603
Age: F2      0.0880506  0.235149 0.374445   0.7081
Age: F3       0.356955  0.247228  1.44383   0.1488
Lrn: SL       0.292138   0.18565  1.57359   0.1156

julia> println("Estimated theta = ", nbrmodel.model.rr.d.r)
Estimated theta = 1.2748930396601978

```

## Other examples

An example of a simple linear model in R is
```r
> coef(summary(lm(optden ~ carb, Formaldehyde)))
               Estimate  Std. Error    t value     Pr(>|t|)
(Intercept) 0.005085714 0.007833679  0.6492115 5.515953e-01
carb        0.876285714 0.013534536 64.7444207 3.409192e-07
```
The corresponding model with the `GLM` package is

```jldoctest
julia> using GLM, RDatasets

julia> form = dataset("datasets", "Formaldehyde")
6×2 DataFrames.DataFrame
│ Row │ Carb │ OptDen │
├─────┼──────┼────────┤
│ 1   │ 0.1  │ 0.086  │
│ 2   │ 0.3  │ 0.269  │
│ 3   │ 0.5  │ 0.446  │
│ 4   │ 0.6  │ 0.538  │
│ 5   │ 0.7  │ 0.626  │
│ 6   │ 0.9  │ 0.782  │

julia> lm1 = fit(LinearModel, @formula(OptDen ~ Carb), form)
StatsModels.DataFrameRegressionModel{GLM.LinearModel{GLM.LmResp{Array{Float64,1}},GLM.DensePredChol{Float64,Base.LinAlg.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}

Formula: OptDen ~ 1 + Carb

Coefficients:
               Estimate  Std.Error  t value Pr(>|t|)
(Intercept)  0.00508571 0.00783368 0.649211   0.5516
Carb           0.876286  0.0135345  64.7444    <1e-6

julia> confint(lm1)
2×2 Array{Float64,2}:
 -0.0166641  0.0268355
  0.838708   0.913864

```

A more complex example in R is
```r
> coef(summary(lm(sr ~ pop15 + pop75 + dpi + ddpi, LifeCycleSavings)))
                 Estimate   Std. Error    t value     Pr(>|t|)
(Intercept) 28.5660865407 7.3545161062  3.8841558 0.0003338249
pop15       -0.4611931471 0.1446422248 -3.1885098 0.0026030189
pop75       -1.6914976767 1.0835989307 -1.5609998 0.1255297940
dpi         -0.0003369019 0.0009311072 -0.3618293 0.7191731554
ddpi         0.4096949279 0.1961971276  2.0881801 0.0424711387
```
with the corresponding Julia code
```jldoctest
julia> LifeCycleSavings = dataset("datasets", "LifeCycleSavings")
50×6 DataFrames.DataFrame
│ Row │ Country        │ SR    │ Pop15 │ Pop75 │ DPI     │ DDPI  │
├─────┼────────────────┼───────┼───────┼───────┼─────────┼───────┤
│ 1   │ Australia      │ 11.43 │ 29.35 │ 2.87  │ 2329.68 │ 2.87  │
│ 2   │ Austria        │ 12.07 │ 23.32 │ 4.41  │ 1507.99 │ 3.93  │
│ 3   │ Belgium        │ 13.17 │ 23.8  │ 4.43  │ 2108.47 │ 3.82  │
│ 4   │ Bolivia        │ 5.75  │ 41.89 │ 1.67  │ 189.13  │ 0.22  │
│ 5   │ Brazil         │ 12.88 │ 42.19 │ 0.83  │ 728.47  │ 4.56  │
│ 6   │ Canada         │ 8.79  │ 31.72 │ 2.85  │ 2982.88 │ 2.43  │
│ 7   │ Chile          │ 0.6   │ 39.74 │ 1.34  │ 662.86  │ 2.67  │
│ 8   │ China          │ 11.9  │ 44.75 │ 0.67  │ 289.52  │ 6.51  │
⋮
│ 42  │ Tunisia        │ 2.81  │ 46.12 │ 1.21  │ 249.87  │ 1.13  │
│ 43  │ United Kingdom │ 7.81  │ 23.27 │ 4.46  │ 1813.93 │ 2.01  │
│ 44  │ United States  │ 7.56  │ 29.81 │ 3.43  │ 4001.89 │ 2.45  │
│ 45  │ Venezuela      │ 9.22  │ 46.4  │ 0.9   │ 813.39  │ 0.53  │
│ 46  │ Zambia         │ 18.56 │ 45.25 │ 0.56  │ 138.33  │ 5.14  │
│ 47  │ Jamaica        │ 7.72  │ 41.12 │ 1.73  │ 380.47  │ 10.23 │
│ 48  │ Uruguay        │ 9.24  │ 28.13 │ 2.72  │ 766.54  │ 1.88  │
│ 49  │ Libya          │ 8.89  │ 43.69 │ 2.07  │ 123.58  │ 16.71 │
│ 50  │ Malaysia       │ 4.71  │ 47.2  │ 0.66  │ 242.69  │ 5.08  │

julia> fm2 = fit(LinearModel, @formula(SR ~ Pop15 + Pop75 + DPI + DDPI), LifeCycleSavings)
StatsModels.DataFrameRegressionModel{GLM.LinearModel{GLM.LmResp{Array{Float64,1}},GLM.DensePredChol{Float64,Base.LinAlg.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}

Formula: SR ~ 1 + Pop15 + Pop75 + DPI + DDPI

Coefficients:
                 Estimate   Std.Error   t value Pr(>|t|)
(Intercept)       28.5661     7.35452   3.88416   0.0003
Pop15           -0.461193    0.144642  -3.18851   0.0026
Pop75             -1.6915      1.0836    -1.561   0.1255
DPI          -0.000336902 0.000931107 -0.361829   0.7192
DDPI             0.409695    0.196197   2.08818   0.0425
```

The `glm` function (or equivalently, `fit(GeneralizedLinearModel, ...)`)
works similarly to the R `glm` function except that the `family`
argument is replaced by a `Distribution` type and, optionally, a `Link` type.
The first example from `?glm` in R is

```r
glm> ## Dobson (1990) Page 93: Randomized Controlled Trial :
glm> counts <- c(18,17,15,20,10,20,25,13,12)

glm> outcome <- gl(3,1,9)

glm> treatment <- gl(3,3)

glm> print(d.AD <- data.frame(treatment, outcome, counts))
  treatment outcome counts
1         1       1     18
2         1       2     17
3         1       3     15
4         2       1     20
5         2       2     10
6         2       3     20
7         3       1     25
8         3       2     13
9         3       3     12

glm> glm.D93 <- glm(counts ~ outcome + treatment, family=poisson())

glm> anova(glm.D93)
Analysis of Deviance Table

Model: poisson, link: log

Response: counts

Terms added sequentially (first to last)


          Df Deviance Resid. Df Resid. Dev
NULL                          8    10.5814
outcome    2   5.4523         6     5.1291
treatment  2   0.0000         4     5.1291

glm> ## No test:
glm> summary(glm.D93)

Call:
glm(formula = counts ~ outcome + treatment, family = poisson())

Deviance Residuals:
       1         2         3         4         5         6         7         8  
-0.67125   0.96272  -0.16965  -0.21999  -0.95552   1.04939   0.84715  -0.09167  
       9  
-0.96656  

Coefficients:
              Estimate Std. Error z value Pr(>|z|)    
(Intercept)  3.045e+00  1.709e-01  17.815   <2e-16 ***
outcome2    -4.543e-01  2.022e-01  -2.247   0.0246 *  
outcome3    -2.930e-01  1.927e-01  -1.520   0.1285    
treatment2   3.795e-16  2.000e-01   0.000   1.0000    
treatment3   3.553e-16  2.000e-01   0.000   1.0000    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for poisson family taken to be 1)

    Null deviance: 10.5814  on 8  degrees of freedom
Residual deviance:  5.1291  on 4  degrees of freedom
AIC: 56.761

Number of Fisher Scoring iterations: 4
```
In Julia this becomes
```jldoctest
julia> using DataFrames, CategoricalArrays, GLM

julia> dobson = DataFrame(Counts    = [18.,17,15,20,10,20,25,13,12],
                          Outcome   = categorical([1,2,3,1,2,3,1,2,3]),
                          Treatment = categorical([1,1,1,2,2,2,3,3,3]))
9×3 DataFrames.DataFrame
│ Row │ Counts │ Outcome │ Treatment │
├─────┼────────┼─────────┼───────────┤
│ 1   │ 18.0   │ 1       │ 1         │
│ 2   │ 17.0   │ 2       │ 1         │
│ 3   │ 15.0   │ 3       │ 1         │
│ 4   │ 20.0   │ 1       │ 2         │
│ 5   │ 10.0   │ 2       │ 2         │
│ 6   │ 20.0   │ 3       │ 2         │
│ 7   │ 25.0   │ 1       │ 3         │
│ 8   │ 13.0   │ 2       │ 3         │
│ 9   │ 12.0   │ 3       │ 3         │


julia> gm1 = fit(GeneralizedLinearModel, @formula(Counts ~ Outcome + Treatment), dobson, Poisson())
StatsModels.DataFrameRegressionModel{GLM.GeneralizedLinearModel{GLM.GlmResp{Array{Float64,1},Distributions.Poisson{Float64},GLM.LogLink},GLM.DensePredChol{Float64,Base.LinAlg.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}

Formula: Counts ~ 1 + Outcome + Treatment

Coefficients:
                 Estimate Std.Error     z value Pr(>|z|)
(Intercept)       3.04452  0.170899     17.8148   <1e-70
Outcome: 2      -0.454255  0.202171    -2.24689   0.0246
Outcome: 3      -0.292987  0.192742     -1.5201   0.1285
Treatment: 2  4.61065e-16       0.2 2.30532e-15   1.0000
Treatment: 3  3.44687e-17       0.2 1.72344e-16   1.0000

julia> deviance(gm1)
5.129141077001145
```

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

## API

### Types defined in the package

```@docs
LinearModel
LmResp
LinPred
GlmResp
DensePredQR
DensePredChol
```

### Constructors for models

The most general approach to fitting a model is with the [`fit`](@ref) function, as in
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

### Methods for model updating
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

### Links and methods applied to them
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
