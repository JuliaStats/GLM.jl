# Examples

```@meta
DocTestSetup = quote
    using CategoricalArrays, DataFrames, Distributions, GLM, RDatasets, Optim
end
```

## Linear regression
```jldoctest
julia> using DataFrames, GLM, StatsBase

julia> data = DataFrame(X=[1,2,3], Y=[2,4,7])
3×2 DataFrame
 Row │ X      Y
     │ Int64  Int64
─────┼──────────────
   1 │     1      2
   2 │     2      4
   3 │     3      7

julia> ols = lm(@formula(Y ~ X), data)
LinearModel

Y ~ 1 + X

Coefficients:
─────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error      t  Pr(>|t|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
(Intercept)  -0.666667    0.62361   -1.07    0.4788   -8.59038    7.25704
X             2.5         0.288675   8.66    0.0732   -1.16797    6.16797
─────────────────────────────────────────────────────────────────────────

julia> round.(stderror(ols), digits=5)
2-element Vector{Float64}:
 0.62361
 0.28868

julia> round.(predict(ols), digits=5)
3-element Vector{Float64}:
 1.83333
 4.33333
 6.83333

julia> round.(confint(ols); digits=5)
2×2 Matrix{Float64}:
 -8.59038  7.25704
 -1.16797  6.16797

julia> round(r2(ols); digits=5)
0.98684

julia> round(adjr2(ols); digits=5)
0.97368

julia> round(deviance(ols); digits=5)
0.16667

julia> dof(ols)
3

julia> dof_residual(ols)
1.0

julia> round(aic(ols); digits=5)
5.84252

julia> round(aicc(ols); digits=5)
-18.15748

julia> round(bic(ols); digits=5)
3.13835

julia> round(dispersion(ols.model); digits=5)
0.40825

julia> round(loglikelihood(ols); digits=5)
0.07874

julia> round(nullloglikelihood(ols); digits=5)
-6.41736

julia> round.(vcov(ols); digits=5)
2×2 Matrix{Float64}:
  0.38889  -0.16667
 -0.16667   0.08333
```
By default, the `lm` method uses the Cholesky factorization which is known as fast but numerically unstable, especially for ill-conditioned design matrices. Also, the Cholesky method aggressively detects multicollinearity. You can use the `method` keyword argument to apply a more stable QR factorization method.

```jldoctest
julia> data = DataFrame(X=[1,2,3], Y=[2,4,7]);

julia> ols = lm(@formula(Y ~ X), data; method=:qr)
LinearModel

Y ~ 1 + X

Coefficients:
─────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error      t  Pr(>|t|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
(Intercept)  -0.666667    0.62361   -1.07    0.4788   -8.59038    7.25704
X             2.5         0.288675   8.66    0.0732   -1.16797    6.16797
─────────────────────────────────────────────────────────────────────────
```
The following example shows that QR decomposition works better for an ill-conditioned design matrix. The linear model with the QR method is a better model than the linear model with Cholesky decomposition method since the estimated loglikelihood of previous model is higher.
Note that, the condition number of the design matrix is quite high (≈ 3.52e7).

```
julia> X = [-0.4011512997627107 0.6368622664511552;
            -0.0808472925693535 0.12835204623364604;
            -0.16931095045225217 0.2687956795496601;
            -0.4110745650568839 0.6526163576003452;
            -0.4035951747670475 0.6407421349445884;
            -0.4649907741370211 0.7382129928076485;
            -0.15772708898883683 0.25040532268222715;
            -0.38144358562952446 0.6055745630707645;
            -0.1012787681395544 0.16078875117643368;
            -0.2741403589052255 0.4352214984054432];

julia> y = [4.362866166172215,
            0.8792840060172619,
            1.8414020451091684,
            4.470790758717395,
            4.3894454833815395,
            5.0571760643993455,
            1.7154177874916376,
            4.148527704012107,
            1.1014936742570425,
            2.9815131910316097];

julia> modelqr = lm(X, y; method=:qr)
LinearModel

Coefficients:
────────────────────────────────────────────────────────────────
       Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────
x1   5.00389   0.0560164   89.33    <1e-12    4.87472    5.13307
x2  10.0025    0.035284   283.48    <1e-16    9.92109   10.0838
────────────────────────────────────────────────────────────────

julia> modelchol = lm(X, y; method=:cholesky)
LinearModel

Coefficients:
────────────────────────────────────────────────────────────────
       Coef.  Std. Error       t  Pr(>|t|)  Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────
x1   5.17647   0.0849184   60.96    <1e-11    4.98065    5.37229
x2  10.1112    0.053489   189.03    <1e-15    9.98781   10.2345
────────────────────────────────────────────────────────────────

julia> loglikelihood(modelqr) > loglikelihood(modelchol)
true
```
Since the Cholesky method with `dropcollinear = true` aggressively detects multicollinearity,
if you ever encounter multicollinearity in any GLM model with Cholesky,
it is worth trying the same model with QR decomposition.
The following example is taken from `Introductory Econometrics: A Modern Approach, 7e" by Jeffrey M. Wooldridge`.
The dataset is used to study the relationship between firm size—often measured by annual sales—and spending on
research and development (R&D).
The following shows that for the given model,
the Cholesky method detects multicollinearity in the design matrix with `dropcollinear=true`
and hence does not estimate all parameters as opposed to QR.

```jldoctest
julia> y = [9.42190647125244, 2.084805727005, 3.9376676082611, 2.61976027488708, 4.04761934280395, 2.15384602546691,
            2.66240668296813, 4.39475727081298, 5.74520826339721, 3.59616208076477, 1.54265284538269, 2.59368276596069,
            1.80476510524749, 1.69270837306976, 3.04201245307922, 2.18389105796813, 2.73844122886657, 2.88134002685546,
            2.46666669845581, 3.80616021156311, 5.12149810791015, 6.80378007888793, 3.73669862747192, 1.21332454681396,
            2.54629635810852, 5.1612901687622, 1.86798071861267, 1.21465551853179, 6.31019830703735, 1.02669405937194, 
            2.50623273849487, 1.5936255455017];

julia> x = [4570.2001953125, 2830, 596.799987792968, 133.600006103515, 42, 390, 93.9000015258789, 907.900024414062,
            19773, 39709, 2936.5, 2513.80004882812, 1124.80004882812, 921.599975585937, 2432.60009765625, 6754,
            1066.30004882812, 3199.89990234375, 150, 509.700012207031, 1452.69995117187, 8995, 1212.30004882812,
            906.599975585937, 2592, 201.5, 2617.80004882812, 502.200012207031, 2824, 292.200012207031, 7621, 1631.5];

julia> rdchem = DataFrame(rdintens=y, sales=x);

julia> mdl = lm(@formula(rdintens ~ sales + sales^2), rdchem; method=:cholesky)
LinearModel

rdintens ~ 1 + sales + :(sales ^ 2)

Coefficients:
───────────────────────────────────────────────────────────────────────────────────────
                    Coef.     Std. Error       t  Pr(>|t|)      Lower 95%     Upper 95%
───────────────────────────────────────────────────────────────────────────────────────
(Intercept)   0.0          NaN            NaN       NaN     NaN            NaN
sales         0.000852509    0.000156784    5.44    <1e-05    0.000532313    0.00117271
sales ^ 2    -1.97385e-8     4.56287e-9    -4.33    0.0002   -2.90571e-8    -1.04199e-8
───────────────────────────────────────────────────────────────────────────────────────

julia> mdl = lm(@formula(rdintens ~ sales + sales^2), rdchem; method=:qr)
LinearModel

rdintens ~ 1 + sales + :(sales ^ 2)

Coefficients:
─────────────────────────────────────────────────────────────────────────────────
                    Coef.   Std. Error      t  Pr(>|t|)    Lower 95%    Upper 95%
─────────────────────────────────────────────────────────────────────────────────
(Intercept)   2.61251      0.429442      6.08    <1e-05   1.73421     3.49082
sales         0.000300571  0.000139295   2.16    0.0394   1.56805e-5  0.000585462
sales ^ 2    -6.94594e-9   3.72614e-9   -1.86    0.0725  -1.45667e-8  6.7487e-10
─────────────────────────────────────────────────────────────────────────────────
```


## Probit regression
```jldoctest
julia> data = DataFrame(X=[1,2,2], Y=[1,0,1])
3×2 DataFrame
 Row │ X      Y
     │ Int64  Int64
─────┼──────────────
   1 │     1      1
   2 │     2      0
   3 │     2      1

julia> probit = glm(@formula(Y ~ X), data, Binomial(), ProbitLink())
GeneralizedLinearModel

Y ~ 1 + X

Coefficients:
────────────────────────────────────────────────────────────────────────
                Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────────────
(Intercept)   9.63839     293.909   0.03    0.9738   -566.414    585.69
X            -4.81919     146.957  -0.03    0.9738   -292.849    283.211
────────────────────────────────────────────────────────────────────────
```

## Negative binomial regression
```jldoctest
julia> using GLM, RDatasets

julia> quine = dataset("MASS", "quine")
146×5 DataFrame
 Row │ Eth   Sex   Age   Lrn   Days
     │ Cat…  Cat…  Cat…  Cat…  Int32
─────┼───────────────────────────────
   1 │ A     M     F0    SL        2
   2 │ A     M     F0    SL       11
   3 │ A     M     F0    SL       14
   4 │ A     M     F0    AL        5
   5 │ A     M     F0    AL        5
   6 │ A     M     F0    AL       13
   7 │ A     M     F0    AL       20
   8 │ A     M     F0    AL       22
  ⋮  │  ⋮     ⋮     ⋮     ⋮      ⋮
 140 │ N     F     F3    AL        3
 141 │ N     F     F3    AL        3
 142 │ N     F     F3    AL        5
 143 │ N     F     F3    AL       15
 144 │ N     F     F3    AL       18
 145 │ N     F     F3    AL       22
 146 │ N     F     F3    AL       37
                     131 rows omitted

julia> nbrmodel = glm(@formula(Days ~ Eth+Sex+Age+Lrn), quine, NegativeBinomial(2.0), LogLink())
GeneralizedLinearModel

Days ~ 1 + Eth + Sex + Age + Lrn

Coefficients:
────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)   Lower 95%   Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)   2.88645      0.227144  12.71    <1e-36   2.44125     3.33164
Eth: N       -0.567515     0.152449  -3.72    0.0002  -0.86631    -0.26872
Sex: M        0.0870771    0.159025   0.55    0.5840  -0.224606    0.398761
Age: F1      -0.445076     0.239087  -1.86    0.0627  -0.913678    0.0235251
Age: F2       0.0927999    0.234502   0.40    0.6923  -0.366816    0.552416
Age: F3       0.359485     0.246586   1.46    0.1449  -0.123814    0.842784
Lrn: SL       0.296768     0.185934   1.60    0.1105  -0.0676559   0.661191
────────────────────────────────────────────────────────────────────────────

julia> nbrmodel = negbin(@formula(Days ~ Eth+Sex+Age+Lrn), quine, LogLink())
GeneralizedLinearModel

Days ~ 1 + Eth + Sex + Age + Lrn

Coefficients:
────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)   Lower 95%   Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)   2.89453      0.227415  12.73    <1e-36   2.4488      3.34025
Eth: N       -0.569341     0.152656  -3.73    0.0002  -0.868541   -0.270141
Sex: M        0.0823881    0.159209   0.52    0.6048  -0.229655    0.394431
Age: F1      -0.448464     0.238687  -1.88    0.0603  -0.916281    0.0193536
Age: F2       0.0880506    0.235149   0.37    0.7081  -0.372834    0.548935
Age: F3       0.356955     0.247228   1.44    0.1488  -0.127602    0.841513
Lrn: SL       0.292138     0.18565    1.57    0.1156  -0.0717297   0.656006
────────────────────────────────────────────────────────────────────────────

julia> println("Estimated theta = ", round(nbrmodel.rr.d.r, digits=5))
Estimated theta = 1.27489

```

## Julia and R comparisons

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
```jldoctest
julia> LifeCycleSavings = dataset("datasets", "LifeCycleSavings")
50×6 DataFrame
 Row │ Country         SR       Pop15    Pop75    DPI      DDPI
     │ String15        Float64  Float64  Float64  Float64  Float64
─────┼─────────────────────────────────────────────────────────────
   1 │ Australia         11.43    29.35     2.87  2329.68     2.87
   2 │ Austria           12.07    23.32     4.41  1507.99     3.93
   3 │ Belgium           13.17    23.8      4.43  2108.47     3.82
   4 │ Bolivia            5.75    41.89     1.67   189.13     0.22
   5 │ Brazil            12.88    42.19     0.83   728.47     4.56
   6 │ Canada             8.79    31.72     2.85  2982.88     2.43
   7 │ Chile              0.6     39.74     1.34   662.86     2.67
   8 │ China             11.9     44.75     0.67   289.52     6.51
  ⋮  │       ⋮            ⋮        ⋮        ⋮        ⋮        ⋮
  44 │ United States      7.56    29.81     3.43  4001.89     2.45
  45 │ Venezuela          9.22    46.4      0.9    813.39     0.53
  46 │ Zambia            18.56    45.25     0.56   138.33     5.14
  47 │ Jamaica            7.72    41.12     1.73   380.47    10.23
  48 │ Uruguay            9.24    28.13     2.72   766.54     1.88
  49 │ Libya              8.89    43.69     2.07   123.58    16.71
  50 │ Malaysia           4.71    47.2      0.66   242.69     5.08
                                                    35 rows omitted

julia> fm2 = fit(LinearModel, @formula(SR ~ Pop15 + Pop75 + DPI + DDPI), LifeCycleSavings)
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
```jldoctest
julia> using DataFrames, CategoricalArrays, GLM

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

julia> gm1 = fit(GeneralizedLinearModel, @formula(Counts ~ Outcome + Treatment), dobson, Poisson())
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

## Linear regression with PowerLink

In this example, we choose the best model from a set of λs, based on minimum BIC.

```jldoctest; filter = r"(\d*)\.(\d{6})\d+" => s"\1.\2"
julia> using GLM, RDatasets, StatsBase, DataFrames, Optim

julia> trees = DataFrame(dataset("datasets", "trees"))
31×3 DataFrame
 Row │ Girth    Height  Volume  
     │ Float64  Int64   Float64 
─────┼──────────────────────────
   1 │     8.3      70     10.3
   2 │     8.6      65     10.3
   3 │     8.8      63     10.2
   4 │    10.5      72     16.4
   5 │    10.7      81     18.8
   6 │    10.8      83     19.7
   7 │    11.0      66     15.6
   8 │    11.0      75     18.2
  ⋮  │    ⋮       ⋮        ⋮
  25 │    16.3      77     42.6
  26 │    17.3      81     55.4
  27 │    17.5      82     55.7
  28 │    17.9      80     58.3
  29 │    18.0      80     51.5
  30 │    18.0      80     51.0
  31 │    20.6      87     77.0
                 16 rows omitted
                 
julia> bic_glm(λ) = bic(glm(@formula(Volume ~ Height + Girth), trees, Normal(), PowerLink(λ)));

julia> optimal_bic = optimize(bic_glm, -1.0, 1.0);

julia> round(optimal_bic.minimizer, digits = 5) # Optimal λ
0.40935

julia> glm(@formula(Volume ~ Height + Girth), trees, Normal(), PowerLink(optimal_bic.minimizer)) # Best model
GeneralizedLinearModel

Volume ~ 1 + Height + Girth

Coefficients:
────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)   Lower 95%   Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)  -1.07586    0.352543    -3.05    0.0023  -1.76684    -0.384892
Height        0.0232172  0.00523331   4.44    <1e-05   0.0129601   0.0334743
Girth         0.242837   0.00922555  26.32    <1e-99   0.224756    0.260919
────────────────────────────────────────────────────────────────────────────

julia> round(optimal_bic.minimum, digits=5)
156.37638
```