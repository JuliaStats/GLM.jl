# Linear models (lm's) and generalized linear models (glm's) in Julia

## Installation

This package requires Steve Johnson's
[NLopt](https://github.com/stevengj/NLopt.jl.git) package for
Julia. Before installing the `NLopt` package be sure to read the
installation instructions as it requires you to have installed the
`nlopt` library of C functions.

Once the `NLopt` package is installed

```julia
Pkg.add("GLM")
```

will install this package.

The `GLM` package also depends on the `DataFrames`, `Distributions`
and `NumericExtensions` packages.

The `RDatasets` package is useful for fitting models to compare with
the results from R.

## Methods applied to fitted models

Many of the methods provided by this package have names similar to those in R.
- `coef`: extract the estimates of the coefficients in the model
- `deviance`: measure of the model fit, weighted residual sum of squares for lm's
- `df_residual`: degrees of freedom for residuals, when meaningful
- `glm`: fit a generalized linear model
- `lm`: fit a linear model
- `stderr`: standard errors of the coefficients
- `vcov`: estimated variance-covariance matrix of the coefficient estimates


An example of a simple linear model in `R` is
```R
> coef(summary(lm(optden ~ carb, Formaldehyde)))
               Estimate  Std. Error    t value     Pr(>|t|)
(Intercept) 0.005085714 0.007833679  0.6492115 5.515953e-01
carb        0.876285714 0.013534536 64.7444207 3.409192e-07
```
The corresponding model with the `GLM` package is

```julia
julia> using RDatasets, GLM

julia> form = data("datasets", "Formaldehyde")
6x2 DataFrame:
        carb optden
[1,]     0.1  0.086
[2,]     0.3  0.269
[3,]     0.5  0.446
[4,]     0.6  0.538
[5,]     0.7  0.626
[6,]     0.9  0.782

julia> fm1 = lm(:(optden ~ carb), form)

Formula: optden ~ carb

Coefficients:
2x4 DataFrame:
          Estimate  Std.Error  t value   Pr(>|t|)
[1,]    0.00508571 0.00783368 0.649211   0.551595
[2,]      0.876286  0.0135345  64.7444 3.40919e-7

julia> confint(fm1)
2x2 Float64 Array:
 -0.0166641  0.0268355
  0.838708   0.913864 
```

A more complex example in `R` is
```R
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
julia> LifeCycleSavings = data("datasets", "LifeCycleSavings");

julia> fm2 = lm(:(sr ~ pop15 + pop75 + dpi + ddpi), LifeCycleSavings)

Formula: sr ~ :(+(pop15,pop75,dpi,ddpi))

Coefficients:
5x4 DataFrame:
            Estimate   Std.Error   t value    Pr(>|t|)
[1,]         28.5661     7.35452   3.88416 0.000333825
[2,]       -0.461193    0.144642  -3.18851  0.00260302
[3,]         -1.6915      1.0836    -1.561     0.12553
[4,]    -0.000336902 0.000931107 -0.361829    0.719173
[5,]        0.409695    0.196197   2.08818   0.0424711

```

The `glm` function works similarly to the corresponding R function
except that the `family` argument is replaced by a `Distribution` type
and, optionally, a `Link` type.  The first example from `?glm` in R is
```R
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
julia> dobson = DataFrame({[18.,17,15,20,10,20,25,13,12], gl(3,1,9), gl(3,3)},
                                 ["counts","outcome","treatment"])
9x3 DataFrame:
        counts outcome treatment
[1,]      18.0       1         1
[2,]      17.0       2         1
[3,]      15.0       3         1
[4,]      20.0       1         2
[5,]      10.0       2         2
[6,]      20.0       3         2
[7,]      25.0       1         3
[8,]      13.0       2         3
[9,]      12.0       3         3


julia> fm3 = glm(:(counts ~ outcome + treatment), dobson, Poisson())
Formula: counts ~ :(+(outcome,treatment))

Coefficients:
5x4 DataFrame:
          Estimate Std.Error    z value    Pr(>|z|)
[1,]       3.04452  0.170893    17.8153 5.37376e-71
[2,]     -0.454255  0.202152    -2.2471   0.0246337
[3,]     -0.292987  0.192728   -1.52021    0.128457
[4,]    1.92349e-8  0.199983 9.61826e-8         1.0
[5,]    8.38339e-9  0.199986 4.19198e-8         1.0

julia> deviance(fm3)    # does not agree with the value from R
46.76131840195778
```

Typical distributions for use with glm and their canonical link
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

## Fitting linear mixed-effects models

The `lmm` function is similar to the function of the same name in the
[lme4](http://cran.R-project.org/package=lme4) package for
[R](http://www.R-project.org).  The first two arguments for in the `R`
version are `formula` and `data`.  The principle method for the
`Julia` version takes these arguments.

### A model fit to the `Dyestuff` data from the `lme4` package

The simplest example of a mixed-effects model that we use in the
[lme4 package for R](https://github.com/lme4/lme4) is a model fit to
the `Dyestuff` data.

```R
> str(Dyestuff)
'data.frame':	30 obs. of  2 variables:
 $ Batch: Factor w/ 6 levels "A","B","C","D",..: 1 1 1 1 1 2 2 2 2 2 ...
 $ Yield: num  1545 1440 1440 1520 1580 ...
> (fm1 <- lmer(Yield ~ 1|Batch, Dyestuff, REML=FALSE))
Linear mixed model fit by maximum likelihood ['lmerMod']
Formula: Yield ~ 1 | Batch 
   Data: Dyestuff 

      AIC       BIC    logLik  deviance 
 333.3271  337.5307 -163.6635  327.3271 

Random effects:
 Groups   Name        Variance Std.Dev.
 Batch    (Intercept) 1388     37.26   
 Residual             2451     49.51   
Number of obs: 30, groups: Batch, 6

Fixed effects:
            Estimate Std. Error t value
(Intercept)  1527.50      17.69   86.33
```

These `Dyestuff` data are available in the `RDatasets` package for `julia`
```julia
julia> using MixedModels, RDatasets

julia> ds = data("lme4","Dyestuff");

julia> dump(ds)
DataFrame  30 observations of 2 variables
  Batch: PooledDataArray{ASCIIString,Uint8,1}(30) ["A", "A", "A", "A"]
  Yield: DataArray{Float64,1}(30) [1545.0, 1440.0, 1440.0, 1520.0]
```

The main difference from `R` in a simple call to `lmm` is the need to
pass the formula as an expression, which means enclosing it in `:()`.
Also, `lmm` defaults to maximum likelihood estimates.

```julia
julia> fm1 = lmm(:(Yield ~ 1|Batch), ds)
Linear mixed model fit by maximum likelihood
 logLik: -163.6635299406109, deviance: 327.3270598812218

  Variance components:
    Std. deviation scale:[37.26047449632836,49.51007020929394]
    Variance scale:[1388.342959691536,2451.2470521292157]
  Number of obs: 30; levels of grouping factors:[6]

  Fixed-effects parameters:
        Estimate Std.Error z value
[1,]      1527.5   17.6946 86.3258
```

(At present the formatting of the output is less than wonderful.)

Optionally the model can fit through an explicit call to the `fit`
function, which may take a second argument indicating a verbose fit.

```julia
julia> m = fit(lmm(Formula(:(Yield ~ 1|Batch)), ds; dofit=false),true);
f_1: 327.7670216246145, [1.0]
f_2: 331.0361932224437, [1.75]
f_3: 330.6458314144857, [0.25]
f_4: 327.69511270610866, [0.97619]
f_5: 327.56630914532184, [0.928569]
f_6: 327.3825965130752, [0.833327]
f_7: 327.3531545408492, [0.807188]
f_8: 327.34662982410276, [0.799688]
f_9: 327.34100192001785, [0.792188]
f_10: 327.33252535370985, [0.777188]
f_11: 327.32733056112147, [0.747188]
f_12: 327.3286190977697, [0.739688]
f_13: 327.32706023603697, [0.752777]
f_14: 327.3270681545395, [0.753527]
f_15: 327.3270598812218, [0.752584]
FTOL_REACHED
```

The numeric representation of the model has type
```julia
julia> typeof(m)
LMMGeneral{Int32}
```

Those familiar with the `lme4` package for `R` will see the usual
suspects.
```julia
julia> fixef(m)
1-element Float64 Array:
 1527.5

julia> ranef(m)
1-element Array{Float64,2} Array:
 1x6 Float64 Array:
 -16.6283  0.369517  26.9747  -21.8015  53.5799  -42.4944

julia> ranef(m,true)  # on the U scale
1-element Array{Float64,2} Array:
 1x6 Float64 Array:
 -22.0949  0.490998  35.8428  -28.9689  71.1947  -56.4647

julia> deviance(m)
327.3270598812218
```

## A more substantial example

Fitting a model to the `Dyestuff` data is trivial.  The `InstEval`
data in the `lme4` package is more of a challenge in that there are
nearly 75,000 evaluations by 2972 students on a total of 1128
instructors.

```julia
julia> inst = data("lme4","InstEval");

julia> dump(inst)
DataFrame  73421 observations of 7 variables
  s: PooledDataArray{ASCIIString,Uint16,1}(73421) ["1", "1", "1", "1"]
  d: PooledDataArray{ASCIIString,Uint16,1}(73421) ["1002", "1050", "1582", "2050"]
  studage: PooledDataArray{ASCIIString,Uint8,1}(73421) ["2", "2", "2", "2"]
  lectage: PooledDataArray{ASCIIString,Uint8,1}(73421) ["2", "1", "2", "2"]
  service: PooledDataArray{ASCIIString,Uint8,1}(73421) ["0", "1", "0", "1"]
  dept: PooledDataArray{ASCIIString,Uint8,1}(73421) ["2", "6", "2", "3"]
  y: DataArray{Int32,1}(73421) [5, 2, 5, 3]

julia> @time fm2 = lmm(:(y ~ dept*service + (1|s) + (1|d)), inst)
elapsed time: 8.862889736 seconds (434911572 bytes allocated)
Linear mixed model fit by maximum likelihood
 logLik: -118792.777, deviance: 237585.553

  Variance components:
    Std. deviation scale:{0.32467999999999997,0.50835,1.1767}
    Variance scale:{0.10541999999999999,0.25842,1.3847}
  Number of obs: 73421; levels of grouping factors:[2972,1128]

  Fixed-effects parameters:
           Estimate Std.Error   z value
[1,]        3.22961  0.064053   50.4209
[2,]       0.129536  0.101294   1.27882
[3,]      -0.176751 0.0881352  -2.00545
[4,]      0.0517102 0.0817524  0.632522
[5,]      0.0347319  0.085621  0.405647
[6,]        0.14594 0.0997984   1.46235
[7,]       0.151689 0.0816897   1.85689
[8,]       0.104206  0.118751  0.877517
[9,]      0.0440401 0.0962985  0.457329
[10,]     0.0517546 0.0986029  0.524879
[11,]     0.0466719  0.101942  0.457828
[12,]     0.0563461 0.0977925   0.57618
[13,]     0.0596536  0.100233   0.59515
[14,]    0.00556281  0.110867 0.0501757
[15,]      0.252025 0.0686507   3.67112
[16,]     -0.180757  0.123179  -1.46744
[17,]     0.0186492  0.110017  0.169512
[18,]     -0.282269 0.0792937  -3.55979
[19,]     -0.494464 0.0790278  -6.25683
[20,]     -0.392054  0.110313  -3.55403
[21,]     -0.278547 0.0823727  -3.38154
[22,]     -0.189526  0.111449  -1.70056
[23,]     -0.499868 0.0885423  -5.64553
[24,]     -0.497162 0.0917162  -5.42065
[25,]      -0.24042 0.0982071   -2.4481
[26,]     -0.223013 0.0890548  -2.50422
[27,]     -0.516997 0.0809077  -6.38997
[28,]     -0.384773  0.091843  -4.18946
```

Models with vector-valued random effects can be fit
```julia
julia> sleep = data("lme4","sleepstudy");

julia> dump(sleep)
DataFrame  180 observations of 3 variables
  Reaction: DataArray{Float64,1}(180) [249.56, 258.705, 250.801, 321.44]
  Days: DataArray{Float64,1}(180) [0.0, 1.0, 2.0, 3.0]
  Subject: PooledDataArray{ASCIIString,Uint8,1}(180) ["308", "308", "308", "308"]

julia> fm3 = lmm(:(Reaction ~ Days + (Days|Subject)), sleep))
Linear mixed model fit by maximum likelihood
 logLik: -875.97, deviance: 1751.939

  Variance components:
    Std. deviation scale:{23.784,5.697900000000001,25.592000000000002}
    Variance scale:{565.69,32.466,654.95}
    Correlations:
{
2x2 Float64 Array:
 1.0        0.0813211
 0.0813211  1.0      }
  Number of obs: 180; levels of grouping factors:[18]

  Fixed-effects parameters:
        Estimate Std.Error z value
[1,]     251.405   6.63212 37.9072
[2,]     10.4673   1.50223 6.96783
```

## ToDo

Well, obviously I need to incorporate names for the fixed-effects
coefficients and create a coefficient table.

Special cases can be tuned up.  Much more calculation is being done in
the fit for models with a single grouping factor, models with scalar
random-effects terms only, models with strictly nested grouping
factors and models with crossed or nearly crossed grouping factors.

Also, the results of at least `X'X` and `X'y` should be cached for
cases where weights aren't changing.

Incorporating offsets and weights will be important for GLMMs.

Lots of work to be done.
