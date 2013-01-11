# Linear models (lm's) and generalized linear models (glm's) in Julia

To install the package run (in Julia)
```julia
load("pkg")
Pkg.add("GLM")
```

The `GLM` package depends on `DataFrames` and `Distributions`. Typically one loads it with
```julia
using DataFrames, Distributions, GLM
```
to access the functions from all three packages.  The `RDatasets`
package is useful for fitting models to compare with the results from
R.

Many of the methods provided by this package have names similar to those in R.
- `coef`: extract the estimates of the coefficients in the model
- `deviance`: measure of the model fit, weighted residual sum of squares for lm's
- `df_residual`: degrees of freedom for residuals, when meaningful
- `glm`: fit a generalized linear model
- `lm`: fit a linear model
- `stderr`: standard errors of the coefficients
- `vcov`: estimated variance-covariance matrix of the coefficient estimates

Right now there are no `show` methods for the fitted model types but
those will be added soon.  A call to `coeftable(fm)` where `fm` is a
fitted model produces results similar to those of `coef(summary(fm))`
in R.

An example of a simple linear model in R is
```R
> coef(summary(lm(optden ~ carb, Formaldehyde)))
               Estimate  Std. Error    t value     Pr(>|t|)
(Intercept) 0.005085714 0.007833679  0.6492115 5.515953e-01
carb        0.876285714 0.013534536 64.7444207 3.409192e-07
```
The corresponding model with the `GLM` package is

```julia
julia> using RDatasets, GLM

julia> Formaldehyde = data("datasets", "Formaldehyde")
6x3 DataFrame:
          carb optden
[1,]    1  0.1  0.086
[2,]    2  0.3  0.269
[3,]    3  0.5  0.446
[4,]    4  0.6  0.538
[5,]    5  0.7  0.626
[6,]    6  0.9  0.782

julia> fm1 = lm(:(optden ~ carb), Formaldehyde)  # lots of noisy output
...
julia> coeftable(fm1)
2x4 DataFrame:
          Estimate  Std.Error  t value   Pr(>|t|)
[1,]    0.00508571 0.00783368 0.649211   0.551595
[2,]      0.876286  0.0135345  64.7444 3.40919e-7
```

A more complex example in R is
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
julia> LifeCycleSavings = data("datasets", "LifeCycleSavings")
...
julia> fm2 = lm(:(sr ~ pop15 + pop75 + dpi + ddpi), LifeCycleSavings)
...
julia> coeftable(fm2)
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
...
julia> coeftable(fm3)
5x4 DataFrame:
          Estimate Std.Error    z value    Pr(>|z|)
[1,]       3.04452  0.170893    17.8153 5.37376e-71
[2,]     -0.454255  0.202152    -2.2471   0.0246337
[3,]     -0.292987  0.192728   -1.52021    0.128457
[4,]    1.92349e-8  0.199983 9.61826e-8         1.0
[5,]    8.38339e-9  0.199986 4.19198e-8         1.0
julia> deviance(fm3)  # something wrong here - need to check the R definition
46.761318401957794
```

Typical distributions for use with glm and their canonical link
functions are
    Bernoulli (LogitLink)
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


