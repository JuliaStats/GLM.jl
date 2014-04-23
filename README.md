# Linear models (lm's) and generalized linear models (glm's) in Julia

[![Build Status](https://travis-ci.org/JuliaStats/GLM.jl.png)](https://travis-ci.org/JuliaStats/GLM.jl)

## Older versions

This documentation applies to GLM.jl 0.4, which has not yet been released. For documentation of GLM.jl 0.3.2, the latest release for Julia 0.3, see [https://github.com/JuliaStats/GLM.jl/tree/v0.3.2](here). For documentation of GLM.jl 0.2.4, the latest release for Julia 0.2, see [https://github.com/JuliaStats/GLM.jl/tree/v0.2.4](here).

## Installation

```julia
Pkg.add("GLM")
```

will install this package.

The `GLM` package also depends on the `DataFrames`, `Distributions`
and `NumericExtensions` packages.

The `RDatasets` package is useful for fitting models to compare with
the results from R.

## Methods applied to fitted models

Many of the methods provided by this package have names similar to those in [R](http://www.r-project.org).
- `coef`: extract the estimates of the coefficients in the model
- `deviance`: measure of the model fit, weighted residual sum of squares for lm's
- `df_residual`: degrees of freedom for residuals, when meaningful
- `glm`: fit a generalized linear model
- `lm`: fit a linear model
- `stderr`: standard errors of the coefficients
- `vcov`: estimated variance-covariance matrix of the coefficient estimates


An example of a simple linear model in `R` is
```s
> coef(summary(lm(optden ~ carb, Formaldehyde)))
               Estimate  Std. Error    t value     Pr(>|t|)
(Intercept) 0.005085714 0.007833679  0.6492115 5.515953e-01
carb        0.876285714 0.013534536 64.7444207 3.409192e-07
```
The corresponding model with the `GLM` package is

```julia
julia> using GLM, RDatasets

julia> form = dataset("datasets","Formaldehyde")
6x2 DataFrame
|-------|------|--------|
| Row # | Carb | OptDen |
| 1     | 0.1  | 0.086  |
| 2     | 0.3  | 0.269  |
| 3     | 0.5  | 0.446  |
| 4     | 0.6  | 0.538  |
| 5     | 0.7  | 0.626  |
| 6     | 0.9  | 0.782  |

julia> lm1 = fit(LmMod, OptDen ~ Carb, form)
Formula: OptDen ~ Carb

Coefficients:
               Estimate  Std.Error  t value Pr(>|t|)
(Intercept)  0.00508571 0.00783368 0.649211   0.5516
Carb           0.876286  0.0135345  64.7444   3.4e-7


julia> confint(lm1)
2x2 Array{Float64,2}:
 -0.0166641  0.0268355
  0.838708   0.913864 
```

A more complex example in `R` is
```s
> coef(summary(lm(sr ~ pop15 + pop75 + dpi + ddpi, LifeCycleSavings)))
                 Estimate   Std. Error    t value     Pr(>|t|)
(Intercept) 28.5660865407 7.3545161062  3.8841558 0.0003338249
pop15       -0.4611931471 0.1446422248 -3.1885098 0.0026030189
pop75       -1.6914976767 1.0835989307 -1.5609998 0.1255297940
dpi         -0.0003369019 0.0009311072 -0.3618293 0.7191731554
ddpi         0.4096949279 0.1961971276  2.0881801 0.0424711387
```
with the corresponding Julia code
```julia
julia> LifeCycleSavings = dataset("datasets", "LifeCycleSavings")
50x6 DataFrame
|-------|----------------|-------|-------|-------|---------|-------|
| Row # | Country        | SR    | Pop15 | Pop75 | DPI     | DDPI  |
| 1     | Australia      | 11.43 | 29.35 | 2.87  | 2329.68 | 2.87  |
| 2     | Austria        | 12.07 | 23.32 | 4.41  | 1507.99 | 3.93  |
| 3     | Belgium        | 13.17 | 23.8  | 4.43  | 2108.47 | 3.82  |
| 4     | Bolivia        | 5.75  | 41.89 | 1.67  | 189.13  | 0.22  |
| 5     | Brazil         | 12.88 | 42.19 | 0.83  | 728.47  | 4.56  |
| 6     | Canada         | 8.79  | 31.72 | 2.85  | 2982.88 | 2.43  |
| 7     | Chile          | 0.6   | 39.74 | 1.34  | 662.86  | 2.67  |
| 8     | China          | 11.9  | 44.75 | 0.67  | 289.52  | 6.51  |
| 9     | Colombia       | 4.98  | 46.64 | 1.06  | 276.65  | 3.08  |
⋮
| 41    | Turkey         | 5.13  | 43.42 | 1.08  | 389.66  | 2.96  |
| 42    | Tunisia        | 2.81  | 46.12 | 1.21  | 249.87  | 1.13  |
| 43    | United Kingdom | 7.81  | 23.27 | 4.46  | 1813.93 | 2.01  |
| 44    | United States  | 7.56  | 29.81 | 3.43  | 4001.89 | 2.45  |
| 45    | Venezuela      | 9.22  | 46.4  | 0.9   | 813.39  | 0.53  |
| 46    | Zambia         | 18.56 | 45.25 | 0.56  | 138.33  | 5.14  |
| 47    | Jamaica        | 7.72  | 41.12 | 1.73  | 380.47  | 10.23 |
| 48    | Uruguay        | 9.24  | 28.13 | 2.72  | 766.54  | 1.88  |
| 49    | Libya          | 8.89  | 43.69 | 2.07  | 123.58  | 16.71 |
| 50    | Malaysia       | 4.71  | 47.2  | 0.66  | 242.69  | 5.08  |

julia> fm2 = fit(LmMod, SR ~ Pop15 + Pop75 + DPI + DDPI, LifeCycleSavings)
Formula: SR ~ :(+(Pop15,Pop75,DPI,DDPI))

Coefficients:
                 Estimate   Std.Error   t value Pr(>|t|)
(Intercept)       28.5661     7.35452   3.88416  0.00033
Pop15           -0.461193    0.144642  -3.18851   0.0026
Pop75             -1.6915      1.0836    -1.561   0.1255
DPI          -0.000336902 0.000931107 -0.361829   0.7192
DDPI             0.409695    0.196197   2.08818   0.0425
```

The `glm` function works similarly to the corresponding R function
except that the `family` argument is replaced by a `Distribution` type
and, optionally, a `Link` type.  The first example from `?glm` in R is
```s
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
```julia
julia> dobson = DataFrame(Counts = [18.,17,15,20,10,20,25,13,12],
                          Outcome = gl(3,1,9),
                          Treatment = gl(3,3))
9x3 DataFrame
|-------|--------|---------|-----------|
| Row # | Counts | Outcome | Treatment |
| 1     | 18.0   | 1       | 1         |
| 2     | 17.0   | 2       | 1         |
| 3     | 15.0   | 3       | 1         |
| 4     | 20.0   | 1       | 2         |
| 5     | 10.0   | 2       | 2         |
| 6     | 20.0   | 3       | 2         |
| 7     | 25.0   | 1       | 3         |
| 8     | 13.0   | 2       | 3         |
| 9     | 12.0   | 3       | 3         |

julia> gm1 = fit(GlmMod, Counts ~ Outcome + Treatment, dobson, Poisson())
Formula: Counts ~ :(+(Outcome,Treatment))

Coefficients:
                   Estimate Std.Error      z value Pr(>|z|)
(Intercept)         3.04452  0.170899      17.8148  < eps()
Outcome - 2       -0.454255  0.202171     -2.24689   0.0246
Outcome - 3       -0.292987  0.192742      -1.5201   0.1285
Treatment - 2   5.36273e-16       0.2  2.68137e-15      1.0
Treatment - 3  -5.07534e-17       0.2 -2.53767e-16      1.0

julia> deviance(gm1)
5.129141077001149
```

Typical distributions for use with `glm` and their canonical link
functions are

     Binomial (LogitLink)
        Gamma (InverseLink)
       Normal (IdentityLink)
      Poisson (LogLink)

Currently the available Link types are

    CauchitLink
    CloglogLink
    IdentityLink
    InverseLink
    LogitLink
    LogLink
    ProbitLink

Other examples are shown in ```test/glmFit.jl```.

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

