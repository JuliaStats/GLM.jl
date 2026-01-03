# Examples

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
1

julia> round(aic(ols); digits=5)
5.84252

julia> round(aicc(ols); digits=5)
-18.15748

julia> round(bic(ols); digits=5)
3.13835

julia> round(dispersion(ols); digits=5)
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

# Linear regression with Cholesky decomposition

By default, the `lm` method uses the QR factorization method, which is stable
but relatively slow. The `method` keyword argument can be set to use Cholesky method
instead for faster model fitting, but it can be numerically unstable, especially
for ill-conditioned design matrices. Also, the Cholesky method aggressively
detects multicollinearity.

The following example shows that QR decomposition works better for an ill-conditioned
design matrix. The linear model with the QR method is a better model than the
linear model with Cholesky decomposition method since the estimated loglikelihood
of previous model is higher.
Note that the condition number of the design matrix is quite high (≈ 3.52e7).

```jldoctest
julia> using GLM

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
The following example is taken from *Introductory Econometrics: A Modern Approach* (7th edition)
by Jeffrey M. Wooldridge.
The dataset is used to study the relationship between firm size—often measured by annual sales—and spending on
research and development (R&D).
The following shows that for the given model,
the Cholesky method detects multicollinearity in the design matrix with `dropcollinear=true`
and hence does not estimate all parameters as opposed to QR.

```jldoctest
julia> using GLM, DataFrames

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
julia> using GLM, DataFrames

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

## Linear regression with PowerLink

In this example, we choose the best model from a set of λs, based on minimum BIC.

```jldoctest; filter = r"(\d*)\.(\d{6})\d+" => s"\1.\2"
julia> using GLM, RDatasets, StatsBase, DataFrames, Optim

julia> trees = DataFrame(dataset("datasets", "trees"));

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
Girth         0.242837   0.00922556  26.32    <1e-99   0.224756    0.260919
────────────────────────────────────────────────────────────────────────────

julia> round(optimal_bic.minimum, digits=5)
156.37638
```
