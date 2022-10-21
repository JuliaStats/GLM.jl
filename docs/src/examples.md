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
StatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}

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
StatsModels.TableRegressionModel{GeneralizedLinearModel{GLM.GlmResp{Vector{Float64}, Binomial{Float64}, ProbitLink}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}

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
StatsModels.TableRegressionModel{GeneralizedLinearModel{GLM.GlmResp{Vector{Float64}, NegativeBinomial{Float64}, LogLink}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}

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
StatsModels.TableRegressionModel{GeneralizedLinearModel{GLM.GlmResp{Vector{Float64}, NegativeBinomial{Float64}, LogLink}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}

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

julia> println("Estimated theta = ", round(nbrmodel.model.rr.d.r, digits=5))
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
StatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}

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
StatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}

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
StatsModels.TableRegressionModel{GeneralizedLinearModel{GLM.GlmResp{Vector{Float64}, Poisson{Float64}, LogLink}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}

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

```jldoctest
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
StatsModels.TableRegressionModel{GeneralizedLinearModel{GLM.GlmResp{Vector{Float64}, Normal{Float64}, PowerLink}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}

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