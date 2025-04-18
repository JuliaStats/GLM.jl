var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#GLM-Documentation-1",
    "page": "Home",
    "title": "GLM Documentation",
    "category": "section",
    "text": ""
},

{
    "location": "#Package-summary-1",
    "page": "Home",
    "title": "Package summary",
    "category": "section",
    "text": "Linear and generalized linear models in Julia"
},

{
    "location": "manual/#",
    "page": "Manual",
    "title": "Manual",
    "category": "page",
    "text": ""
},

{
    "location": "manual/#Manual-1",
    "page": "Manual",
    "title": "Manual",
    "category": "section",
    "text": ""
},

{
    "location": "manual/#Installation-1",
    "page": "Manual",
    "title": "Installation",
    "category": "section",
    "text": "Pkg.add(\"GLM\")will install this package and its dependencies, which includes the Distributions package.The RDatasets package is useful for fitting models on standard R datasets to compare the results with those from R."
},

{
    "location": "manual/#Fitting-GLM-models-1",
    "page": "Manual",
    "title": "Fitting GLM models",
    "category": "section",
    "text": "To fit a Generalized Linear Model (GLM), use the function, glm(formula, data, family, link), where,formula: uses column symbols from the DataFrame data, for example, if names(data)=[:Y,:X1,:X2], then a valid formula is @formula(Y ~ X1 + X2)\ndata: a DataFrame which may contain NA values, any rows with NA values are ignored\nfamily: chosen from Bernoulli(), Binomial(), Gamma(), Normal(), Poisson(), or NegativeBinomial(θ)\nlink: chosen from the list below, for example, LogitLink() is a valid link for the Binomial() familyTypical distributions for use with glm and their canonical link functions are       Bernoulli (LogitLink)\n        Binomial (LogitLink)\n           Gamma (InverseLink)\n InverseGaussian (InverseSquareLink)\nNegativeBinomial (LogLink)\n          Normal (IdentityLink)\n         Poisson (LogLink)Currently the available Link types areCauchitLink\nCloglogLink\nIdentityLink\nInverseLink\nInverseSquareLink\nLogitLink\nLogLink\nNegativeBinomialLink\nProbitLink\nSqrtLinkThe NegativeBinomial distribution belongs to the exponential family only if θ (the shape parameter) is fixed, thus θ has to be provided if we use glm with NegativeBinomial family. If one would like to also estimate θ, then negbin(formula, data, link) should be used instead.An intercept is included in any GLM by default."
},

{
    "location": "manual/#Methods-applied-to-fitted-models-1",
    "page": "Manual",
    "title": "Methods applied to fitted models",
    "category": "section",
    "text": "Many of the methods provided by this package have names similar to those in R.coef: extract the estimates of the coefficients in the model\ndeviance: measure of the model fit, weighted residual sum of squares for lm\'s\ndof_residual: degrees of freedom for residuals, when meaningful\nglm: fit a generalized linear model (an alias for fit(GeneralizedLinearModel, ...))\nlm: fit a linear model (an alias for fit(LinearModel, ...))\nstderror: standard errors of the coefficients\nvcov: estimated variance-covariance matrix of the coefficient estimates\npredict : obtain predicted values of the dependent variable from the fitted modelNote that the canonical link for negative binomial regression is NegativeBinomialLink, but in practice one typically uses LogLink."
},

{
    "location": "manual/#Separation-of-response-object-and-predictor-object-1",
    "page": "Manual",
    "title": "Separation of response object and predictor object",
    "category": "section",
    "text": "The general approach in this code is to separate functionality related to the response from that related to the linear predictor.  This allows for greater generality by mixing and matching different subtypes of the abstract type LinPred and the abstract type ModResp.A LinPred type incorporates the parameter vector and the model matrix.  The parameter vector is a dense numeric vector but the model matrix can be dense or sparse.  A LinPred type must incorporate some form of a decomposition of the weighted model matrix that allows for the solution of a system X\'W * X * delta=X\'wres where W is a diagonal matrix of \"X weights\", provided as a vector of the square roots of the diagonal elements, and wres is a weighted residual vector.Currently there are two dense predictor types, DensePredQR and DensePredChol, and the usual caveats apply.  The Cholesky version is faster but somewhat less accurate than that QR version. The skeleton of a distributed predictor type is in the code but not yet fully fleshed out.  Because Julia by default uses OpenBLAS, which is already multi-threaded on multicore machines, there may not be much advantage in using distributed predictor types.A ModResp type must provide methods for the wtres and sqrtxwts generics.  Their values are the arguments to the updatebeta methods of the LinPred types.  The Float64 value returned by updatedelta is the value of the convergence criterion.Similarly, LinPred types must provide a method for the linpred generic.  In general linpred takes an instance of a LinPred type and a step factor.  Methods that take only an instance of a LinPred type use a default step factor of 1.  The value of linpred is the argument to the updatemu method for ModResp types.  The updatemu method returns the updated deviance."
},

{
    "location": "examples/#",
    "page": "Examples",
    "title": "Examples",
    "category": "page",
    "text": ""
},

{
    "location": "examples/#Examples-1",
    "page": "Examples",
    "title": "Examples",
    "category": "section",
    "text": "DocTestSetup = quote\n    using CategoricalArrays, DataFrames, Distributions, GLM, RDatasets\nend"
},

{
    "location": "examples/#Linear-regression-1",
    "page": "Examples",
    "title": "Linear regression",
    "category": "section",
    "text": "julia> using DataFrames, GLM\n\njulia> data = DataFrame(X=[1,2,3], Y=[2,4,7])\n3×2 DataFrames.DataFrame\n│ Row │ X     │ Y     │\n│     │ Int64 │ Int64 │\n├─────┼───────┼───────┤\n│ 1   │ 1     │ 2     │\n│ 2   │ 2     │ 4     │\n│ 3   │ 3     │ 7     │\n\njulia> ols = lm(@formula(Y ~ X), data)\nStatsModels.DataFrameRegressionModel{LinearModel{LmResp{Array{Float64,1}},DensePredChol{Float64,LinearAlgebra.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}\n\nFormula: Y ~ 1 + X\n\nCoefficients:\n              Estimate Std.Error  t value Pr(>|t|)\n(Intercept)  -0.666667   0.62361 -1.06904   0.4788\nX                  2.5  0.288675  8.66025   0.0732\n\njulia> round.(stderror(ols), digits=5)\n2-element Array{Float64,1}:\n 0.62361\n 0.28868\n\njulia> round.(predict(ols), digits=5)\n3-element Array{Float64,1}:\n 1.83333\n 4.33333\n 6.83333."
},

{
    "location": "examples/#Probit-regression-1",
    "page": "Examples",
    "title": "Probit regression",
    "category": "section",
    "text": "julia> data = DataFrame(X=[1,2,2], Y=[1,0,1])\n3×2 DataFrames.DataFrame\n│ Row │ X     │ Y     │\n│     │ Int64 │ Int64 │\n├─────┼───────┼───────┤\n│ 1   │ 1     │ 1     │\n│ 2   │ 2     │ 0     │\n│ 3   │ 2     │ 1     │\n\njulia> probit = glm(@formula(Y ~ X), data, Binomial(), ProbitLink())\nStatsModels.DataFrameRegressionModel{GeneralizedLinearModel{GlmResp{Array{Float64,1},Binomial{Float64},ProbitLink},DensePredChol{Float64,LinearAlgebra.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}\n\nFormula: Y ~ 1 + X\n\nCoefficients:\n             Estimate Std.Error    z value Pr(>|z|)\n(Intercept)   9.63839   293.909  0.0327937   0.9738\nX            -4.81919   146.957 -0.0327933   0.9738"
},

{
    "location": "examples/#Negative-binomial-regression-1",
    "page": "Examples",
    "title": "Negative binomial regression",
    "category": "section",
    "text": "julia> using GLM, RDatasets\n\njulia> quine = dataset(\"MASS\", \"quine\")\n146×5 DataFrames.DataFrame\n│ Row │ Eth          │ Sex          │ Age          │ Lrn          │ Days  │\n│     │ Categorical… │ Categorical… │ Categorical… │ Categorical… │ Int32 │\n├─────┼──────────────┼──────────────┼──────────────┼──────────────┼───────┤\n│ 1   │ A            │ M            │ F0           │ SL           │ 2     │\n│ 2   │ A            │ M            │ F0           │ SL           │ 11    │\n│ 3   │ A            │ M            │ F0           │ SL           │ 14    │\n│ 4   │ A            │ M            │ F0           │ AL           │ 5     │\n│ 5   │ A            │ M            │ F0           │ AL           │ 5     │\n│ 6   │ A            │ M            │ F0           │ AL           │ 13    │\n│ 7   │ A            │ M            │ F0           │ AL           │ 20    │\n⋮\n│ 139 │ N            │ F            │ F3           │ AL           │ 22    │\n│ 140 │ N            │ F            │ F3           │ AL           │ 3     │\n│ 141 │ N            │ F            │ F3           │ AL           │ 3     │\n│ 142 │ N            │ F            │ F3           │ AL           │ 5     │\n│ 143 │ N            │ F            │ F3           │ AL           │ 15    │\n│ 144 │ N            │ F            │ F3           │ AL           │ 18    │\n│ 145 │ N            │ F            │ F3           │ AL           │ 22    │\n│ 146 │ N            │ F            │ F3           │ AL           │ 37    │\n\njulia> nbrmodel = glm(@formula(Days ~ Eth+Sex+Age+Lrn), quine, NegativeBinomial(2.0), LogLink())\nStatsModels.DataFrameRegressionModel{GeneralizedLinearModel{GlmResp{Array{Float64,1},NegativeBinomial{Float64},LogLink},DensePredChol{Float64,LinearAlgebra.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}\n\nFormula: Days ~ 1 + Eth + Sex + Age + Lrn\n\nCoefficients:\n              Estimate Std.Error  z value Pr(>|z|)\n(Intercept)    2.88645  0.227144  12.7076   <1e-36\nEth: N       -0.567515  0.152449 -3.72265   0.0002\nSex: M       0.0870771  0.159025 0.547568   0.5840\nAge: F1      -0.445076  0.239087 -1.86157   0.0627\nAge: F2      0.0927999  0.234502 0.395731   0.6923\nAge: F3       0.359485  0.246586  1.45785   0.1449\nLrn: SL       0.296768  0.185934  1.59609   0.1105\n\njulia> nbrmodel = negbin(@formula(Days ~ Eth+Sex+Age+Lrn), quine, LogLink())\nStatsModels.DataFrameRegressionModel{GeneralizedLinearModel{GlmResp{Array{Float64,1},NegativeBinomial{Float64},LogLink},DensePredChol{Float64,LinearAlgebra.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}\n\nFormula: Days ~ 1 + Eth + Sex + Age + Lrn\n\nCoefficients:\n              Estimate Std.Error  z value Pr(>|z|)\n(Intercept)    2.89453  0.227415   12.728   <1e-36\nEth: N       -0.569341  0.152656 -3.72957   0.0002\nSex: M       0.0823881  0.159209 0.517485   0.6048\nAge: F1      -0.448464  0.238687 -1.87888   0.0603\nAge: F2      0.0880506  0.235149 0.374445   0.7081\nAge: F3       0.356955  0.247228  1.44383   0.1488\nLrn: SL       0.292138   0.18565  1.57359   0.1156\n\njulia> println(\"Estimated theta = \", round(nbrmodel.model.rr.d.r, digits=5))\nEstimated theta = 1.27489\n"
},

{
    "location": "examples/#Julia-and-R-comparisons-1",
    "page": "Examples",
    "title": "Julia and R comparisons",
    "category": "section",
    "text": "An example of a simple linear model in R is> coef(summary(lm(optden ~ carb, Formaldehyde)))\n               Estimate  Std. Error    t value     Pr(>|t|)\n(Intercept) 0.005085714 0.007833679  0.6492115 5.515953e-01\ncarb        0.876285714 0.013534536 64.7444207 3.409192e-07The corresponding model with the GLM package isjulia> using GLM, RDatasets\n\njulia> form = dataset(\"datasets\", \"Formaldehyde\")\n6×2 DataFrames.DataFrame\n│ Row │ Carb     │ OptDen   │\n│     │ Float64⍰ │ Float64⍰ │\n├─────┼──────────┼──────────┤\n│ 1   │ 0.1      │ 0.086    │\n│ 2   │ 0.3      │ 0.269    │\n│ 3   │ 0.5      │ 0.446    │\n│ 4   │ 0.6      │ 0.538    │\n│ 5   │ 0.7      │ 0.626    │\n│ 6   │ 0.9      │ 0.782    │\n\njulia> lm1 = fit(LinearModel, @formula(OptDen ~ Carb), form)\nStatsModels.DataFrameRegressionModel{LinearModel{LmResp{Array{Float64,1}},DensePredChol{Float64,LinearAlgebra.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}\n\nFormula: OptDen ~ 1 + Carb\n\nCoefficients:\n               Estimate  Std.Error  t value Pr(>|t|)\n(Intercept)  0.00508571 0.00783368 0.649211   0.5516\nCarb           0.876286  0.0135345  64.7444    <1e-6\n\njulia> confint(lm1)\n2×2 Array{Float64,2}:\n -0.0166641  0.0268355\n  0.838708   0.913864\nA more complex example in R is> coef(summary(lm(sr ~ pop15 + pop75 + dpi + ddpi, LifeCycleSavings)))\n                 Estimate   Std. Error    t value     Pr(>|t|)\n(Intercept) 28.5660865407 7.3545161062  3.8841558 0.0003338249\npop15       -0.4611931471 0.1446422248 -3.1885098 0.0026030189\npop75       -1.6914976767 1.0835989307 -1.5609998 0.1255297940\ndpi         -0.0003369019 0.0009311072 -0.3618293 0.7191731554\nddpi         0.4096949279 0.1961971276  2.0881801 0.0424711387with the corresponding Julia codejulia> LifeCycleSavings = dataset(\"datasets\", \"LifeCycleSavings\")\n50×6 DataFrames.DataFrame\n│ Row │ Country        │ SR       │ Pop15    │ Pop75    │ DPI      │ DDPI     │\n│     │ String⍰        │ Float64⍰ │ Float64⍰ │ Float64⍰ │ Float64⍰ │ Float64⍰ │\n├─────┼────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤\n│ 1   │ Australia      │ 11.43    │ 29.35    │ 2.87     │ 2329.68  │ 2.87     │\n│ 2   │ Austria        │ 12.07    │ 23.32    │ 4.41     │ 1507.99  │ 3.93     │\n│ 3   │ Belgium        │ 13.17    │ 23.8     │ 4.43     │ 2108.47  │ 3.82     │\n│ 4   │ Bolivia        │ 5.75     │ 41.89    │ 1.67     │ 189.13   │ 0.22     │\n│ 5   │ Brazil         │ 12.88    │ 42.19    │ 0.83     │ 728.47   │ 4.56     │\n│ 6   │ Canada         │ 8.79     │ 31.72    │ 2.85     │ 2982.88  │ 2.43     │\n│ 7   │ Chile          │ 0.6      │ 39.74    │ 1.34     │ 662.86   │ 2.67     │\n⋮\n│ 43  │ United Kingdom │ 7.81     │ 23.27    │ 4.46     │ 1813.93  │ 2.01     │\n│ 44  │ United States  │ 7.56     │ 29.81    │ 3.43     │ 4001.89  │ 2.45     │\n│ 45  │ Venezuela      │ 9.22     │ 46.4     │ 0.9      │ 813.39   │ 0.53     │\n│ 46  │ Zambia         │ 18.56    │ 45.25    │ 0.56     │ 138.33   │ 5.14     │\n│ 47  │ Jamaica        │ 7.72     │ 41.12    │ 1.73     │ 380.47   │ 10.23    │\n│ 48  │ Uruguay        │ 9.24     │ 28.13    │ 2.72     │ 766.54   │ 1.88     │\n│ 49  │ Libya          │ 8.89     │ 43.69    │ 2.07     │ 123.58   │ 16.71    │\n│ 50  │ Malaysia       │ 4.71     │ 47.2     │ 0.66     │ 242.69   │ 5.08     │\n\njulia> fm2 = fit(LinearModel, @formula(SR ~ Pop15 + Pop75 + DPI + DDPI), LifeCycleSavings)\nStatsModels.DataFrameRegressionModel{LinearModel{LmResp{Array{Float64,1}},DensePredChol{Float64,LinearAlgebra.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}\n\nFormula: SR ~ 1 + Pop15 + Pop75 + DPI + DDPI\n\nCoefficients:\n                 Estimate   Std.Error   t value Pr(>|t|)\n(Intercept)       28.5661     7.35452   3.88416   0.0003\nPop15           -0.461193    0.144642  -3.18851   0.0026\nPop75             -1.6915      1.0836    -1.561   0.1255\nDPI          -0.000336902 0.000931107 -0.361829   0.7192\nDDPI             0.409695    0.196197   2.08818   0.0425The glm function (or equivalently, fit(GeneralizedLinearModel, ...)) works similarly to the R glm function except that the family argument is replaced by a Distribution type and, optionally, a Link type. The first example from ?glm in R isglm> ## Dobson (1990) Page 93: Randomized Controlled Trial : (slightly modified)\nglm> counts <- c(18,17,15,20,10,21,25,13,13)\n\nglm> outcome <- gl(3,1,9)\n\nglm> treatment <- gl(3,3)\n\nglm> print(d.AD <- data.frame(treatment, outcome, counts))\n  treatment outcome counts\n1         1       1     18\n2         1       2     17\n3         1       3     15\n4         2       1     20\n5         2       2     10\n6         2       3     21\n7         3       1     25\n8         3       2     13\n9         3       3     13\n\nglm> glm.D93 <- glm(counts ~ outcome + treatment, family=poisson())\n\nglm> anova(glm.D93)\nAnalysis of Deviance Table\n\nModel: poisson, link: log\n\nResponse: counts\n\nTerms added sequentially (first to last)\n\n\n          Df Deviance Resid. Df Resid. Dev\nNULL                          8    10.3928\noutcome    2   5.2622         6     5.1307\ntreatment  2   0.0132         4     5.1175\n\nglm> ## No test:\nglm> summary(glm.D93)\n\nCall:\nglm(formula = counts ~ outcome + treatment, family = poisson())\n\nDeviance Residuals: \n      1        2        3        4        5        6        7        8        9  \n-0.6122   1.0131  -0.2819  -0.2498  -0.9784   1.0777   0.8162  -0.1155  -0.8811  \n\nCoefficients:\n            Estimate Std. Error z value Pr(>|z|)    \n(Intercept)   3.0313     0.1712  17.711   <2e-16 ***\noutcome2     -0.4543     0.2022  -2.247   0.0246 *  \noutcome3     -0.2513     0.1905  -1.319   0.1870    \ntreatment2    0.0198     0.1990   0.100   0.9207    \ntreatment3    0.0198     0.1990   0.100   0.9207    \n---\nSignif. codes:  0 \'***\' 0.001 \'**\' 0.01 \'*\' 0.05 \'.\' 0.1 \' \' 1\n\n(Dispersion parameter for poisson family taken to be 1)\n\n    Null deviance: 10.3928  on 8  degrees of freedom\nResidual deviance:  5.1175  on 4  degrees of freedom\nAIC: 56.877\n\nNumber of Fisher Scoring iterations: 4In Julia this becomesjulia> using DataFrames, CategoricalArrays, GLM\n\njulia> dobson = DataFrame(Counts    = [18.,17,15,20,10,21,25,13,13],\n                          Outcome   = categorical([1,2,3,1,2,3,1,2,3]),\n                          Treatment = categorical([1,1,1,2,2,2,3,3,3]))\n9×3 DataFrames.DataFrame\n│ Row │ Counts  │ Outcome      │ Treatment    │\n│     │ Float64 │ Categorical… │ Categorical… │\n├─────┼─────────┼──────────────┼──────────────┤\n│ 1   │ 18.0    │ 1            │ 1            │\n│ 2   │ 17.0    │ 2            │ 1            │\n│ 3   │ 15.0    │ 3            │ 1            │\n│ 4   │ 20.0    │ 1            │ 2            │\n│ 5   │ 10.0    │ 2            │ 2            │\n│ 6   │ 21.0    │ 3            │ 2            │\n│ 7   │ 25.0    │ 1            │ 3            │\n│ 8   │ 13.0    │ 2            │ 3            │\n│ 9   │ 13.0    │ 3            │ 3            │\n\n\njulia> gm1 = fit(GeneralizedLinearModel, @formula(Counts ~ Outcome + Treatment), dobson, Poisson())\nStatsModels.DataFrameRegressionModel{GeneralizedLinearModel{GlmResp{Array{Float64,1},Poisson{Float64},LogLink},DensePredChol{Float64,LinearAlgebra.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}\n\nFormula: Counts ~ 1 + Outcome + Treatment\n\nCoefficients:\n               Estimate Std.Error   z value Pr(>|z|)\n(Intercept)     3.03128  0.171155   17.7107   <1e-69\nOutcome: 2    -0.454255  0.202171  -2.24689   0.0246\nOutcome: 3    -0.251314  0.190476   -1.3194   0.1870\nTreatment: 2  0.0198026  0.199017 0.0995021   0.9207\nTreatment: 3  0.0198026  0.199017 0.0995021   0.9207\n\njulia> round(deviance(gm1), digits=5)\n5.11746"
},

{
    "location": "api/#",
    "page": "API",
    "title": "API",
    "category": "page",
    "text": ""
},

{
    "location": "api/#API-1",
    "page": "API",
    "title": "API",
    "category": "section",
    "text": "DocTestSetup = quote\n    using CategoricalArrays, DataFrames, Distributions, GLM, RDatasets\nend"
},

{
    "location": "api/#GLM.DensePredChol",
    "page": "API",
    "title": "GLM.DensePredChol",
    "category": "type",
    "text": "DensePredChol{T}\n\nA LinPred type with a dense Cholesky factorization of X\'X\n\nMembers\n\nX: model matrix of size n × p with n ≥ p.  Should be full column rank.\nbeta0: base coefficient vector of length p\ndelbeta: increment to coefficient vector, also of length p\nscratchbeta: scratch vector of length p, used in linpred! method\nchol: a Cholesky object created from X\'X, possibly using row weights.\nscratchm1: scratch Matrix{T} of the same size as X\nscratchm2: scratch Matrix{T} os the same size as X\'X\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.DensePredQR",
    "page": "API",
    "title": "GLM.DensePredQR",
    "category": "type",
    "text": "DensePredQR\n\nA LinPred type with a dense, unpivoted QR decomposition of X\n\nMembers\n\nX: Model matrix of size n × p with n ≥ p.  Should be full column rank.\nbeta0: base coefficient vector of length p\ndelbeta: increment to coefficient vector, also of length p\nscratchbeta: scratch vector of length p, used in linpred! method\nqr: a QRCompactWY object created from X, with optional row weights.\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.GlmResp",
    "page": "API",
    "title": "GLM.GlmResp",
    "category": "type",
    "text": "GlmResp\n\nThe response vector and various derived vectors in a generalized linear model.\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.LinearModel",
    "page": "API",
    "title": "GLM.LinearModel",
    "category": "type",
    "text": "LinearModel\n\nA combination of a LmResp and a LinPred\n\nMembers\n\nrr: a LmResp object\npp: a LinPred object\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.LmResp",
    "page": "API",
    "title": "GLM.LmResp",
    "category": "type",
    "text": "LmResp\n\nEncapsulates the response for a linear model\n\nMembers\n\nmu: current value of the mean response vector or fitted value\noffset: optional offset added to the linear predictor to form mu\nwts: optional vector of prior weights\ny: observed response vector\n\nEither or both offset and wts may be of length 0\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.LinPred",
    "page": "API",
    "title": "GLM.LinPred",
    "category": "type",
    "text": "LinPred\n\nAbstract type representing a linear predictor\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.ModResp",
    "page": "API",
    "title": "GLM.ModResp",
    "category": "type",
    "text": "ModResp\n\nAbstract type representing a model response vector\n\n\n\n\n\n"
},

{
    "location": "api/#Types-defined-in-the-package-1",
    "page": "API",
    "title": "Types defined in the package",
    "category": "section",
    "text": "DensePredChol\nDensePredQR\nGlmResp\nLinearModel\nLmResp\nLinPred\nGLM.ModResp"
},

{
    "location": "api/#Constructors-for-models-1",
    "page": "API",
    "title": "Constructors for models",
    "category": "section",
    "text": "The most general approach to fitting a model is with the fit function, as injulia> using Random;\n\njulia> fit(LinearModel, hcat(ones(10), 1:10), randn(MersenneTwister(12321), 10))\nLinearModel{LmResp{Array{Float64,1}},DensePredChol{Float64,LinearAlgebra.Cholesky{Float64,Array{Float64,2}}}}:\n\nCoefficients:\n      Estimate Std.Error  t value Pr(>|t|)\nx1    0.717436  0.775175 0.925515   0.3818\nx2   -0.152062  0.124931 -1.21717   0.2582This model can also be fit asjulia> using Random;\n\njulia> lm(hcat(ones(10), 1:10), randn(MersenneTwister(12321), 10))\nLinearModel{LmResp{Array{Float64,1}},DensePredChol{Float64,LinearAlgebra.Cholesky{Float64,Array{Float64,2}}}}:\n\nCoefficients:\n      Estimate Std.Error  t value Pr(>|t|)\nx1    0.717436  0.775175 0.925515   0.3818\nx2   -0.152062  0.124931 -1.21717   0.2582"
},

{
    "location": "api/#GLM.cancancel",
    "page": "API",
    "title": "GLM.cancancel",
    "category": "function",
    "text": "cancancel(r::GlmResp{V,D,L})\n\nReturns true if dμ/dη for link L is the variance function for distribution D\n\nWhen L is the canonical link for D the derivative of the inverse link is a multiple of the variance function for D.  If they are the same a numerator and denominator term in the expression for the working weights will cancel.\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.delbeta!",
    "page": "API",
    "title": "GLM.delbeta!",
    "category": "function",
    "text": "delbeta!(p::LinPred, r::Vector)\n\nEvaluate and return p.delbeta the increment to the coefficient vector from residual r\n\n\n\n\n\n"
},

{
    "location": "api/#StatsBase.deviance",
    "page": "API",
    "title": "StatsBase.deviance",
    "category": "function",
    "text": "deviance(obj::LinearModel)\n\nFor linear models, the deviance is equal to the residual sum of squares (RSS).\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.dispersion",
    "page": "API",
    "title": "GLM.dispersion",
    "category": "function",
    "text": "dispersion(m::AbstractGLM, sqr::Bool=false)\n\nReturn the estimated dispersion (or scale) parameter for a model\'s distribution, generally written σ² for linear models and ϕ for generalized linear models. It is, by definition, equal to 1 for the Bernoulli, Binomial, and Poisson families.\n\nIf sqr is true, the squared dispersion parameter is returned.\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.installbeta!",
    "page": "API",
    "title": "GLM.installbeta!",
    "category": "function",
    "text": "installbeta!(p::LinPred, f::Real=1.0)\n\nInstall pbeta0 .+= f * p.delbeta and zero out p.delbeta.  Return the updated p.beta0.\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.issubmodel",
    "page": "API",
    "title": "GLM.issubmodel",
    "category": "function",
    "text": "A helper function to determine if mod1 is nested in mod2\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.linpred!",
    "page": "API",
    "title": "GLM.linpred!",
    "category": "function",
    "text": "linpred!(out, p::LinPred, f::Real=1.0)\n\nOverwrite out with the linear predictor from p with factor f\n\nThe effective coefficient vector, p.scratchbeta, is evaluated as p.beta0 .+ f * p.delbeta, and out is updated to p.X * p.scratchbeta\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.linpred",
    "page": "API",
    "title": "GLM.linpred",
    "category": "function",
    "text": "linpred(p::LinPred, f::Read=1.0)\n\nReturn the linear predictor p.X * (p.beta0 .+ f * p.delbeta)\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.lm",
    "page": "API",
    "title": "GLM.lm",
    "category": "function",
    "text": "lm(X, y, allowrankdeficient::Bool=false)\n\nAn alias for fit(LinearModel, X, y, allowrankdeficient)\n\nThe arguments X and y can be a Matrix and a Vector or a Formula and a DataFrame.\n\n\n\n\n\n"
},

{
    "location": "api/#StatsBase.nobs",
    "page": "API",
    "title": "StatsBase.nobs",
    "category": "function",
    "text": "nobs(obj::LinearModel)\nnobs(obj::GLM)\n\nFor linear and generalized linear models, returns the number of rows, or, when prior weights are specified, the sum of weights.\n\n\n\n\n\n"
},

{
    "location": "api/#StatsBase.nulldeviance",
    "page": "API",
    "title": "StatsBase.nulldeviance",
    "category": "function",
    "text": "nulldeviance(obj::LinearModel)\n\nFor linear models, the deviance of the null model is equal to the total sum of squares (TSS).\n\n\n\n\n\n"
},

{
    "location": "api/#StatsBase.predict",
    "page": "API",
    "title": "StatsBase.predict",
    "category": "function",
    "text": "predict(mm::LinearModel, newx::AbstractMatrix;\n        interval::Union{Symbol,Nothing} = nothing, level::Real = 0.95)\n\nIf interval is nothing (the default), return a vector with the predicted values for model mm and new data newx. Otherwise, return a 3-column matrix with the prediction and the lower and upper confidence bounds for a given level (0.95 equates alpha = 0.05). Valid values of interval are :confidence delimiting the  uncertainty of the predicted relationship, and :prediction delimiting estimated bounds for new data points.\n\n\n\n\n\npredict(mm::AbstractGLM, newX::AbstractMatrix; offset::FPVector=Vector{eltype(newX)}(0))\n\nForm the predicted response of model mm from covariate values newX and, optionally, an offset.\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.updateμ!",
    "page": "API",
    "title": "GLM.updateμ!",
    "category": "function",
    "text": "updateμ!{T<:FPVector}(r::GlmResp{T}, linPr::T)\n\nUpdate the mean, working weights and working residuals, in r given a value of the linear predictor, linPr.\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.wrkresp",
    "page": "API",
    "title": "GLM.wrkresp",
    "category": "function",
    "text": "wrkresp(r::GlmResp)\n\nThe working response, r.eta + r.wrkresid - r.offset.\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.wrkresp!",
    "page": "API",
    "title": "GLM.wrkresp!",
    "category": "function",
    "text": "wrkresp!{T<:FPVector}(v::T, r::GlmResp{T})\n\nOverwrite v with the working response of r\n\n\n\n\n\n"
},

{
    "location": "api/#Model-methods-1",
    "page": "API",
    "title": "Model methods",
    "category": "section",
    "text": "GLM.cancancel\ndelbeta!\nStatsBase.deviance\nGLM.dispersion\nGLM.installbeta!\nGLM.issubmodel\nlinpred!\nlinpred\nlm\nStatsBase.nobs\nStatsBase.nulldeviance\nStatsBase.predict\nupdateμ!\nwrkresp\nGLM.wrkresp!"
},

{
    "location": "api/#GLM.Link",
    "page": "API",
    "title": "GLM.Link",
    "category": "type",
    "text": "Link\n\nAn abstract type whose subtypes determine methods for linkfun, linkinv, mueta, and inverselink.\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.Link01",
    "page": "API",
    "title": "GLM.Link01",
    "category": "type",
    "text": "Link01\n\nAn abstract subtype of Link which are links defined on (0, 1)\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.CauchitLink",
    "page": "API",
    "title": "GLM.CauchitLink",
    "category": "type",
    "text": "CauchitLink\n\nA Link01 corresponding to the standard Cauchy distribution, Distributions.Cauchy.\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.CloglogLink",
    "page": "API",
    "title": "GLM.CloglogLink",
    "category": "type",
    "text": "CloglogLink\n\nA Link01 corresponding to the extreme value (or log-Wiebull) distribution.  The link is the complementary log-log transformation, log(1 - log(-μ)).\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.IdentityLink",
    "page": "API",
    "title": "GLM.IdentityLink",
    "category": "type",
    "text": "IdentityLink\n\nThe canonical Link for the Normal distribution, defined as η = μ.\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.InverseLink",
    "page": "API",
    "title": "GLM.InverseLink",
    "category": "type",
    "text": "InverseLink\n\nThe canonical Link for Distributions.Gamma distribution, defined as η = inv(μ).\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.InverseSquareLink",
    "page": "API",
    "title": "GLM.InverseSquareLink",
    "category": "type",
    "text": "InverseSquareLink\n\nThe canonical Link for Distributions.InverseGaussian distribution, defined as η = inv(abs2(μ)).\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.LogitLink",
    "page": "API",
    "title": "GLM.LogitLink",
    "category": "type",
    "text": "LogitLink\n\nThe canonical Link01 for Distributions.Bernoulli and Distributions.Binomial. The inverse link, linkinv, is the c.d.f. of the standard logistic distribution, Distributions.Logistic.\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.LogLink",
    "page": "API",
    "title": "GLM.LogLink",
    "category": "type",
    "text": "LogLink\n\nThe canonical Link for Distributions.Poisson, defined as η = log(μ).\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.NegativeBinomialLink",
    "page": "API",
    "title": "GLM.NegativeBinomialLink",
    "category": "type",
    "text": "NegativeBinomialLink\n\nThe canonical Link for Distributions.NegativeBinomial distribution, defined as η = log(μ/(μ+θ)). The shape parameter θ has to be fixed for the distribution to belong to the exponential family.\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.ProbitLink",
    "page": "API",
    "title": "GLM.ProbitLink",
    "category": "type",
    "text": "ProbitLink\n\nA Link01 whose linkinv is the c.d.f. of the standard normal distribution, Distributions.Normal().\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.SqrtLink",
    "page": "API",
    "title": "GLM.SqrtLink",
    "category": "type",
    "text": "SqrtLink\n\nA Link defined as η = √μ\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.linkfun",
    "page": "API",
    "title": "GLM.linkfun",
    "category": "function",
    "text": "linkfun(L::Link, μ)\n\nReturn η, the value of the linear predictor for link L at mean μ.\n\nExamples\n\njulia> μ = inv(10):inv(5):1\n0.1:0.2:0.9\n\njulia> show(linkfun.(LogitLink(), μ))\n[-2.19722, -0.847298, 0.0, 0.847298, 2.19722]\n\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.linkinv",
    "page": "API",
    "title": "GLM.linkinv",
    "category": "function",
    "text": "linkinv(L::Link, η)\n\nReturn μ, the mean value, for link L at linear predictor value η.\n\nExamples\n\njulia> μ = 0.1:0.2:1\n0.1:0.2:0.9\n\njulia> η = logit.(μ);\n\njulia> linkinv.(LogitLink(), η) ≈ μ\ntrue\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.mueta",
    "page": "API",
    "title": "GLM.mueta",
    "category": "function",
    "text": "mueta(L::Link, η)\n\nReturn the derivative of linkinv, dμ/dη, for link L at linear predictor value η.\n\nExamples\n\njulia> mueta(LogitLink(), 0.0)\n0.25\n\njulia> mueta(CloglogLink(), 0.0) ≈ 0.36787944117144233\ntrue\n\njulia> mueta(LogLink(), 2.0) ≈ 7.38905609893065\ntrue\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.inverselink",
    "page": "API",
    "title": "GLM.inverselink",
    "category": "function",
    "text": "inverselink(L::Link, η)\n\nReturn a 3-tuple of the inverse link, the derivative of the inverse link, and when appropriate, the variance function μ*(1 - μ).\n\nThe variance function is returned as NaN unless the range of μ is (0, 1)\n\nExamples\n\njulia> inverselink(LogitLink(), 0.0)\n(0.5, 0.25, 0.25)\n\njulia> μ, oneminusμ, variance = inverselink(CloglogLink(), 0.0);\n\njulia> μ + oneminusμ ≈ 1\ntrue\n\njulia> μ*(1 - μ) ≈ variance\ntrue\n\njulia> isnan(last(inverselink(LogLink(), 2.0)))\ntrue\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.canonicallink",
    "page": "API",
    "title": "GLM.canonicallink",
    "category": "function",
    "text": "canonicallink(D::Distribution)\n\nReturn the canonical link for distribution D, which must be in the exponential family.\n\nExamples\n\njulia> canonicallink(Bernoulli())\nLogitLink()\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.glmvar",
    "page": "API",
    "title": "GLM.glmvar",
    "category": "function",
    "text": "glmvar(D::Distribution, μ)\n\nReturn the value of the variance function for D at μ\n\nThe variance of D at μ is the product of the dispersion parameter, ϕ, which does not depend on μ and the value of glmvar.  In other words glmvar returns the factor of the variance that depends on μ.\n\nExamples\n\njulia> μ = 1/6:1/3:1;\n\njulia> glmvar.(Normal(), μ)    # constant for Normal()\n3-element Array{Float64,1}:\n 1.0\n 1.0\n 1.0\n\njulia> glmvar.(Bernoulli(), μ) ≈ μ .* (1 .- μ)\ntrue\n\njulia> glmvar.(Poisson(), μ) == μ\ntrue\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.mustart",
    "page": "API",
    "title": "GLM.mustart",
    "category": "function",
    "text": "mustart(D::Distribution, y, wt)\n\nReturn a starting value for μ.\n\nFor some distributions it is appropriate to set μ = y to initialize the IRLS algorithm but for others, notably the Bernoulli, the values of y are not allowed as values of μ and must be modified.\n\nExamples\n\njulia> mustart(Bernoulli(), 0.0, 1) ≈ 1/4\ntrue\n\njulia> mustart(Bernoulli(), 1.0, 1) ≈ 3/4\ntrue\n\njulia> mustart(Binomial(), 0.0, 10) ≈ 1/22\ntrue\n\njulia> mustart(Normal(), 0.0, 1) ≈ 0\ntrue\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.devresid",
    "page": "API",
    "title": "GLM.devresid",
    "category": "function",
    "text": "devresid(D, y, μ)\n\nReturn the squared deviance residual of μ from y for distribution D\n\nThe deviance of a GLM can be evaluated as the sum of the squared deviance residuals.  This is the principal use for these values.  The actual deviance residual, say for plotting, is the signed square root of this value\n\nsign(y - μ) * sqrt(devresid(D, y, μ))\n\nExamples\n\njulia> devresid(Normal(), 0, 0.25) ≈ abs2(0.25)\ntrue\n\njulia> devresid(Bernoulli(), 1, 0.75) ≈ -2*log(0.75)\ntrue\n\njulia> devresid(Bernoulli(), 0, 0.25) ≈ -2*log1p(-0.25)\ntrue\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.dispersion_parameter",
    "page": "API",
    "title": "GLM.dispersion_parameter",
    "category": "function",
    "text": "dispersion_parameter(D)  # not exported\n\nDoes distribution D have a separate dispersion parameter, ϕ?\n\nReturns false for the Bernoulli, Binomial and Poisson distributions, true otherwise.\n\nExamples\n\njulia> show(GLM.dispersion_parameter(Normal()))\ntrue\njulia> show(GLM.dispersion_parameter(Bernoulli()))\nfalse\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.loglik_obs",
    "page": "API",
    "title": "GLM.loglik_obs",
    "category": "function",
    "text": "loglik_obs(D, y, μ, wt, ϕ)  # not exported\n\nReturns wt * logpdf(D(μ, ϕ), y) where the parameters of D are derived from μ and ϕ.\n\nThe wt argument is a multiplier of the result except in the case of the Binomial where wt is the number of trials and μ is the proportion of successes.\n\nThe loglikelihood of a fitted model is the sum of these values over all the observations.\n\n\n\n\n\n"
},

{
    "location": "api/#Links-and-methods-applied-to-them-1",
    "page": "API",
    "title": "Links and methods applied to them",
    "category": "section",
    "text": "Link\nGLM.Link01\nCauchitLink\nCloglogLink\nIdentityLink\nInverseLink\nInverseSquareLink\nLogitLink\nLogLink\nNegativeBinomialLink\nProbitLink\nSqrtLink\nlinkfun\nlinkinv\nmueta\ninverselink\ncanonicallink\nglmvar\nmustart\ndevresid\nGLM.dispersion_parameter\nGLM.loglik_obs"
},

]}
