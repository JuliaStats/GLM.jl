# Comparison with R

GLM.jl is very similar to `lm` and `glm` commands in R and results generally match.
There are some exceptions however, which are documented below.

First let us load packages necessary to run examples:
```jldoctest r-comparison
julia> using CategoricalArrays, CSV, DataFrames, Distributions, GLM, RDatasets, StatsBase
```

## Linear models

An example of a simple linear model in R is
```r
> coef(summary(lm(optden ~ carb, Formaldehyde)))
               Estimate  Std. Error    t value     Pr(>|t|)
(Intercept) 0.005085714 0.007833679  0.6492115 5.515953e-01
carb        0.876285714 0.013534536 64.7444207 3.409192e-07
```
The corresponding model with the `GLM` package is

```jldoctest r-comparison
julia> form = dataset("datasets", "Formaldehyde")
6×2 DataFrame
 Row │ Carb     OptDen  
     │ Float64  Float64 
─────┼──────────────────
   1 │     0.1    0.086
   2 │     0.3    0.269
   3 │     0.5    0.446
   4 │     0.6    0.538
   5 │     0.7    0.626
   6 │     0.9    0.782

julia> lm1 = fit(LinearModel, @formula(OptDen ~ Carb), form)
LinearModel

OptDen ~ 1 + Carb

Coefficients:
───────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      t  Pr(>|t|)   Lower 95%  Upper 95%
───────────────────────────────────────────────────────────────────────────
(Intercept)  0.00508571  0.00783368   0.65    0.5516  -0.0166641  0.0268355
Carb         0.876286    0.0135345   64.74    <1e-06   0.838708   0.913864
───────────────────────────────────────────────────────────────────────────
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
```jldoctest r-comparison
julia> LifeCycleSavings = dataset("datasets", "LifeCycleSavings");

julia> fm2 = lm(@formula(SR ~ Pop15 + Pop75 + DPI + DDPI), LifeCycleSavings)
LinearModel

SR ~ 1 + Pop15 + Pop75 + DPI + DDPI

Coefficients:
─────────────────────────────────────────────────────────────────────────────────
                    Coef.   Std. Error      t  Pr(>|t|)    Lower 95%    Upper 95%
─────────────────────────────────────────────────────────────────────────────────
(Intercept)  28.5661       7.35452       3.88    0.0003  13.7533      43.3788
Pop15        -0.461193     0.144642     -3.19    0.0026  -0.752518    -0.169869
Pop75        -1.6915       1.0836       -1.56    0.1255  -3.87398      0.490983
DPI          -0.000336902  0.000931107  -0.36    0.7192  -0.00221225   0.00153844
DDPI          0.409695     0.196197      2.09    0.0425   0.0145336    0.804856
─────────────────────────────────────────────────────────────────────────────────
```

# Generalized linear models

The `glm` function (or equivalently, `fit(GeneralizedLinearModel, ...)`)
works similarly to the R `glm` function except that the `family`
argument is replaced by a `Distribution` type and, optionally, a `Link` type.
The first example from `?glm` in R is

```r
glm> ## Dobson (1990) Page 93: Randomized Controlled Trial : (slightly modified)
glm> counts <- c(18,17,15,20,10,21,25,13,13)

glm> outcome <- gl(3,1,9)

glm> treatment <- gl(3,3)

glm> print(d.AD <- data.frame(treatment, outcome, counts))
  treatment outcome counts
1         1       1     18
2         1       2     17
3         1       3     15
4         2       1     20
5         2       2     10
6         2       3     21
7         3       1     25
8         3       2     13
9         3       3     13

glm> glm.D93 <- glm(counts ~ outcome + treatment, family=poisson())

glm> anova(glm.D93)
Analysis of Deviance Table

Model: poisson, link: log

Response: counts

Terms added sequentially (first to last)


          Df Deviance Resid. Df Resid. Dev
NULL                          8    10.3928
outcome    2   5.2622         6     5.1307
treatment  2   0.0132         4     5.1175

glm> ## No test:
glm> summary(glm.D93)

Call:
glm(formula = counts ~ outcome + treatment, family = poisson())

Deviance Residuals:
      1        2        3        4        5        6        7        8        9
-0.6122   1.0131  -0.2819  -0.2498  -0.9784   1.0777   0.8162  -0.1155  -0.8811

Coefficients:
            Estimate Std. Error z value Pr(>|z|)
(Intercept)   3.0313     0.1712  17.711   <2e-16 ***
outcome2     -0.4543     0.2022  -2.247   0.0246 *
outcome3     -0.2513     0.1905  -1.319   0.1870
treatment2    0.0198     0.1990   0.100   0.9207
treatment3    0.0198     0.1990   0.100   0.9207
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

(Dispersion parameter for poisson family taken to be 1)

    Null deviance: 10.3928  on 8  degrees of freedom
Residual deviance:  5.1175  on 4  degrees of freedom
AIC: 56.877

Number of Fisher Scoring iterations: 4
```

In Julia this becomes
```jldoctest r-comparison
julia> dobson = DataFrame(Counts    = [18.,17,15,20,10,21,25,13,13],
                          Outcome   = categorical([1,2,3,1,2,3,1,2,3]),
                          Treatment = categorical([1,1,1,2,2,2,3,3,3]))
9×3 DataFrame
 Row │ Counts   Outcome  Treatment 
     │ Float64  Cat…     Cat…      
─────┼─────────────────────────────
   1 │    18.0  1        1
   2 │    17.0  2        1
   3 │    15.0  3        1
   4 │    20.0  1        2
   5 │    10.0  2        2
   6 │    21.0  3        2
   7 │    25.0  1        3
   8 │    13.0  2        3
   9 │    13.0  3        3

julia> gm1 = glm(@formula(Counts ~ Outcome + Treatment), dobson, Poisson())
GeneralizedLinearModel

Counts ~ 1 + Outcome + Treatment

Coefficients:
────────────────────────────────────────────────────────────────────────────
                   Coef.  Std. Error      z  Pr(>|z|)  Lower 95%   Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)    3.03128      0.171155  17.71    <1e-69   2.69582    3.36674
Outcome: 2    -0.454255     0.202171  -2.25    0.0246  -0.850503  -0.0580079
Outcome: 3    -0.251314     0.190476  -1.32    0.1870  -0.624641   0.122012
Treatment: 2   0.0198026    0.199017   0.10    0.9207  -0.370264   0.409869
Treatment: 3   0.0198026    0.199017   0.10    0.9207  -0.370264   0.409869
────────────────────────────────────────────────────────────────────────────

julia> round(deviance(gm1), digits=5)
5.11746
```

Note that for generalized linear models, Wald tests in GLM.jl use the Normal
distribution (z-statistics) by default, while R uses the Student distribution
(t-statistics) for distributions where the dispersion parameter is estimated
(i.e. Normal, inverse Gaussian or Gamma distributions).
The `test` argument to `coeftable` allows choosing between these two options.
This allows obtaining the same p-values using `glm` as those obtained above
with `lm`:
```jldoctest r-comparison
julia> LifeCycleSavings = dataset("datasets", "LifeCycleSavings");

julia> fm2glm = glm(@formula(SR ~ Pop15 + Pop75 + DPI + DDPI), LifeCycleSavings,
                    Normal(), IdentityLink())
GeneralizedLinearModel

SR ~ 1 + Pop15 + Pop75 + DPI + DDPI

Coefficients:
─────────────────────────────────────────────────────────────────────────────────
                    Coef.   Std. Error      z  Pr(>|z|)    Lower 95%    Upper 95%
─────────────────────────────────────────────────────────────────────────────────
(Intercept)  28.5661       7.35452       3.88    0.0001  14.1515      42.9807
Pop15        -0.461193     0.144642     -3.19    0.0014  -0.744687    -0.1777
Pop75        -1.6915       1.0836       -1.56    0.1185  -3.81531      0.432317
DPI          -0.000336902  0.000931107  -0.36    0.7175  -0.00216184   0.00148803
DDPI          0.409695     0.196197      2.09    0.0368   0.0251556    0.794234
─────────────────────────────────────────────────────────────────────────────────

julia> coeftable(fm2glm, test=:t)
─────────────────────────────────────────────────────────────────────────────────
                    Coef.   Std. Error      t  Pr(>|t|)    Lower 95%    Upper 95%
─────────────────────────────────────────────────────────────────────────────────
(Intercept)  28.5661       7.35452       3.88    0.0003  13.7533      43.3788
Pop15        -0.461193     0.144642     -3.19    0.0026  -0.752518    -0.169869
Pop75        -1.6915       1.0836       -1.56    0.1255  -3.87398      0.490983
DPI          -0.000336902  0.000931107  -0.36    0.7192  -0.00221225   0.00153844
DDPI          0.409695     0.196197      2.09    0.0425   0.0145336    0.804856
─────────────────────────────────────────────────────────────────────────────────
```

# Analytic weights

In R, the `weights` argument to `lm` and `glm` takes analytic weights (a.k.a.
inverse-variance weights). For example with a linear regression:
```R
> summary(lm(mpg ~ wt + hp, data = mtcars, weights = 1/wt))

Call:
lm(formula = mpg ~ wt + hp, data = mtcars, weights = 1/wt)

Weighted Residuals:
    Min      1Q  Median      3Q     Max 
-2.2271 -1.0538 -0.3894  0.6397  3.7627 

Coefficients:
             Estimate Std. Error t value Pr(>|t|)    
(Intercept) 39.002317   1.541462  25.302  < 2e-16 ***
wt          -4.443823   0.688300  -6.456 4.59e-07 ***
hp          -0.031460   0.009776  -3.218  0.00317 ** 
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 1.554 on 29 degrees of freedom
Multiple R-squared:  0.8389,	Adjusted R-squared:  0.8278 
F-statistic: 75.49 on 2 and 29 DF,  p-value: 3.189e-12

```

In GLM.jl this can be reproduced by passing an `AnalyticWeights` vector
to the `wts` argument, which can be constructed using `aweights` :
```jldoctest r-comparison
julia> mtcars = dataset("datasets", "mtcars");

julia> lm(@formula(MPG ~ WT + HP), mtcars, wts=aweights(1 ./ mtcars.WT))
LinearModel

MPG ~ 1 + WT + HP

Coefficients:
────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      t  Pr(>|t|)   Lower 95%   Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)  39.0023     1.54146     25.30    <1e-20  35.8497     42.155
WT           -4.44382    0.6883      -6.46    <1e-06  -5.85155    -3.03609
HP           -0.0314601  0.00977604  -3.22    0.0032  -0.0514543  -0.0114658
────────────────────────────────────────────────────────────────────────────
```

In R, weights can also be used to fit logistic regressions on aggregate counts:
```R
> data(UCBAdmissions)
> summary(glm(Admit == "Admitted" ~ Gender + Dept, family=binomial,
              data=as.data.frame(UCBAdmissions), weight=Freq))

Call:
glm(formula = Admit == "Admitted" ~ Gender + Dept, family = binomial, 
    data = as.data.frame(UCBAdmissions), weights = Freq)

Coefficients:
             Estimate Std. Error z value Pr(>|z|)    
(Intercept)   0.58205    0.06899   8.436   <2e-16 ***
GenderFemale  0.09987    0.08085   1.235    0.217    
DeptB        -0.04340    0.10984  -0.395    0.693    
DeptC        -1.26260    0.10663 -11.841   <2e-16 ***
DeptD        -1.29461    0.10582 -12.234   <2e-16 ***
DeptE        -1.73931    0.12611 -13.792   <2e-16 ***
DeptF        -3.30648    0.16998 -19.452   <2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 6044.3  on 23  degrees of freedom
Residual deviance: 5187.5  on 17  degrees of freedom
AIC: 5201.5

Number of Fisher Scoring iterations: 6

```

In GLM.jl, such a logistic regression can be fitted using either
analytic weights or frequency weights (see next section):
```jldoctest r-comparison
julia> ucb = dataset("datasets", "UCBAdmissions");

julia> ucb.AdmitBin = ucb.Admit .== "Admitted";

julia> glm(@formula(AdmitBin ~ Gender + Dept), ucb, Binomial(), wts=aweights(ucb.Freq))
GeneralizedLinearModel

AdmitBin ~ 1 + Gender + Dept

Coefficients:
─────────────────────────────────────────────────────────────────────────────
                   Coef.  Std. Error       z  Pr(>|z|)  Lower 95%   Upper 95%
─────────────────────────────────────────────────────────────────────────────
(Intercept)    0.681921    0.0991106    6.88    <1e-11   0.487668   0.876175
Gender: Male  -0.0998701   0.0808432   -1.24    0.2167  -0.25832    0.0585798
Dept: B       -0.0433979   0.109839    -0.40    0.6928  -0.258678   0.171882
Dept: C       -1.2626      0.106632   -11.84    <1e-31  -1.47159   -1.0536
Dept: D       -1.29461     0.105823   -12.23    <1e-33  -1.50202   -1.0872
Dept: E       -1.73931     0.126113   -13.79    <1e-42  -1.98648   -1.49213
Dept: F       -3.30648     0.169904   -19.46    <1e-83  -3.63949   -2.97347
─────────────────────────────────────────────────────────────────────────────

```

# Frequency weights

Models estimated using `FrequencyWeights` (which can be created using `fweights`,
also known as case weights) in GLM.jl are equivalent to fitting models on a dataset
in which each is repeated a number of times equal to its weight in R.

# Probability weights

Models estimated using `ProbabilityWeights` (which can be created using `pweights`,
also known as sampling weights) generally give the same results as those obtained
using the `survey` R package with a simple design without strata nor clustering.
A few exceptions are mentioned in this section.

A simplified version of the first example from `?svyglm` in R ignoring
strata is:
```R
> library(survey)
> data(api)
> dapi <- svydesign(id=~1, weights=~pw, data=apisrs)
> summary(svyglm(api00 ~ ell + meals + mobility, design=dapi))

Call:
svyglm(formula = api00 ~ ell + meals + mobility, design = dapi)

Survey design:
svydesign(id = ~1, weights = ~pw, data = apisrs)

Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept) 848.0293    18.3336  46.256  < 2e-16 ***
ell          -1.6018     0.6324  -2.533   0.0121 *  
meals        -2.5799     0.4560  -5.658 5.38e-08 ***
mobility     -1.3922     1.2483  -1.115   0.2661    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 6189.033)

Number of Fisher Scoring iterations: 2
```

The same results are obtained with GLM.jl using:
```jldoctest r-comparison
julia> apistrat = dataset("survey", "apisrs");

julia> apistrat.pw = pweights(apistrat.pw);

julia> lm(@formula(api00 ~ ell + meals + mobility), apistrat, wts=apistrat.pw)
LinearModel

api00 ~ 1 + ell + meals + mobility

Coefficients:
──────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error      t  Pr(>|t|)  Lower 95%   Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)  848.029     18.3336    46.26    <1e-99  811.873    884.186
ell           -1.60183    0.632381  -2.53    0.0121   -2.84897   -0.354684
meals         -2.57991    0.456003  -5.66    <1e-07   -3.47921   -1.68061
mobility      -1.39217    1.24835   -1.12    0.2661   -3.85409    1.06974
──────────────────────────────────────────────────────────────────────────
```

A logistic regression model with R's `survey` uses the `quasibinomial` distribution as
the `binomial` prints warnings with non-integer values:
```R
> summary(svyglm(sch.wide~ell+meals+mobility, design=dapi,
                 family=quasibinomial))

Call:
svyglm(formula = sch.wide ~ ell + meals + mobility, design = dapi, 
    family = quasibinomial)

Survey design:
svydesign(id = ~1, weights = ~pw, data = apisrs)

Coefficients:
             Estimate Std. Error t value Pr(>|t|)    
(Intercept)  1.743801   0.463606   3.761 0.000223 ***
ell         -0.021591   0.011493  -1.879 0.061775 .  
meals        0.010966   0.009316   1.177 0.240541    
mobility    -0.014805   0.021964  -0.674 0.501072    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for quasibinomial family taken to be 1.022665)

Number of Fisher Scoring iterations: 4
```

With GLM.jl this model can be estimated simply using the standard `Binomial` distribution:
```jldoctest r-comparison
julia> apistrat.sch_wide_bin = apistrat.sch_wide .== "Yes";

julia> logis = glm(@formula(sch_wide_bin ~ ell + meals + mobility), apistrat,
                   Binomial(), wts=apistrat.pw)
GeneralizedLinearModel

sch_wide_bin ~ 1 + ell + meals + mobility

Coefficients:
──────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)    Lower 95%    Upper 95%
──────────────────────────────────────────────────────────────────────────────
(Intercept)   1.7438     0.463406     3.76    0.0002   0.835542    2.65206
ell          -0.0215908  0.0114877   -1.88    0.0602  -0.0441063   0.000924616
meals         0.0109663  0.00931115   1.18    0.2389  -0.00728325  0.0292158
mobility     -0.0148048  0.0219597   -0.67    0.5002  -0.057845    0.0282353
──────────────────────────────────────────────────────────────────────────────

julia> coeftable(logis, test=:t)
────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      t  Pr(>|t|)    Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)   1.7438     0.463406     3.76    0.0002   0.829899    2.6577
ell          -0.0215908  0.0114877   -1.88    0.0617  -0.0442462   0.0010645
meals         0.0109663  0.00931115   1.18    0.2403  -0.00739664  0.0293292
mobility     -0.0148048  0.0219597   -0.67    0.5010  -0.0581124   0.0285027
────────────────────────────────────────────────────────────────────────────
```