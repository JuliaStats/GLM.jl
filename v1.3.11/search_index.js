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
    "location": "manual/#Categorical-variables-1",
    "page": "Manual",
    "title": "Categorical variables",
    "category": "section",
    "text": "Categorical variables will be dummy coded by default if they are non-numeric or if they are CategoricalVectors within a Tables.jl table (DataFrame, JuliaDB table, named tuple of vectors, etc). Alternatively, you can pass an explicit  contrasts argument if you would like a different contrast coding system or if you are not using DataFrames.The response (dependent) variable may not be categorical.Using a CategoricalVector constructed with categorical or categorical!:julia> using DataFrames, GLM, Random\n\njulia> Random.seed!(1); # Ensure example can be reproduced\n\njulia> data = DataFrame(y = rand(100), x = categorical(repeat([1, 2, 3, 4], 25)));\n\njulia> lm(@formula(y ~ x), data)\nStatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Array{Float64,1}},GLM.DensePredChol{Float64,LinearAlgebra.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}\n\ny ~ 1 + x\n\nCoefficients:\n─────────────────────────────────────────────────────────────────────────────\n              Estimate  Std. Error   t value  Pr(>|t|)   Lower 95%  Upper 95%\n─────────────────────────────────────────────────────────────────────────────\n(Intercept)  0.41335     0.0548456  7.53662     <1e-10   0.304483    0.522218\nx: 2         0.172338    0.0775634  2.2219      0.0286   0.0183756   0.3263  \nx: 3         0.0422104   0.0775634  0.544205    0.5876  -0.111752    0.196172\nx: 4         0.0793591   0.0775634  1.02315     0.3088  -0.074603    0.233321\n─────────────────────────────────────────────────────────────────────────────Using contrasts:julia> data = DataFrame(y = rand(100), x = repeat([1, 2, 3, 4], 25));\n\njulia> lm(@formula(y ~ x), data, contrasts = Dict(:x => DummyCoding()))\nStatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Array{Float64,1}},GLM.DensePredChol{Float64,LinearAlgebra.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}\n\ny ~ 1 + x\n\nCoefficients:\n────────────────────────────────────────────────────────────────────────────────\n               Estimate  Std. Error     t value  Pr(>|t|)   Lower 95%  Upper 95%\n────────────────────────────────────────────────────────────────────────────────\n(Intercept)   0.464446    0.0582412   7.97453      <1e-11   0.348838    0.580054\nx: 2         -0.0057872   0.0823655  -0.0702624    0.9441  -0.169281    0.157707\nx: 3          0.0923976   0.0823655   1.1218       0.2647  -0.0710966   0.255892\nx: 4          0.115145    0.0823655   1.39797      0.1653  -0.0483494   0.278639\n────────────────────────────────────────────────────────────────────────────────"
},

{
    "location": "manual/#Methods-applied-to-fitted-models-1",
    "page": "Manual",
    "title": "Methods applied to fitted models",
    "category": "section",
    "text": "Many of the methods provided by this package have names similar to those in R.coef: extract the estimates of the coefficients in the model\ndeviance: measure of the model fit, weighted residual sum of squares for lm\'s\ndof_residual: degrees of freedom for residuals, when meaningful\nglm: fit a generalized linear model (an alias for fit(GeneralizedLinearModel, ...))\nlm: fit a linear model (an alias for fit(LinearModel, ...))\nr2: R² of a linear model or pseudo-R² of a generalized linear model\nstderror: standard errors of the coefficients\nvcov: estimated variance-covariance matrix of the coefficient estimates\npredict : obtain predicted values of the dependent variable from the fitted modelNote that the canonical link for negative binomial regression is NegativeBinomialLink, but in practice one typically uses LogLink."
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
    "text": "julia> using DataFrames, GLM\n\njulia> data = DataFrame(X=[1,2,3], Y=[2,4,7])\n3×2 DataFrames.DataFrame\n│ Row │ X     │ Y     │\n│     │ Int64 │ Int64 │\n├─────┼───────┼───────┤\n│ 1   │ 1     │ 2     │\n│ 2   │ 2     │ 4     │\n│ 3   │ 3     │ 7     │\n\njulia> ols = lm(@formula(Y ~ X), data)\nStatsModels.DataFrameRegressionModel{LinearModel{LmResp{Array{Float64,1}},DensePredChol{Float64,LinearAlgebra.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}\n\nFormula: Y ~ 1 + X\n\nCoefficients:\n─────────────────────────────────────────────────────────────────────────\n                 Coef.  Std. Error      t  Pr(>|t|)  Lower 95%  Upper 95%\n─────────────────────────────────────────────────────────────────────────\n(Intercept)  -0.666667    0.62361   -1.07    0.4788   -8.59038    7.25704\nX             2.5         0.288675   8.66    0.0732   -1.16797    6.16797\n─────────────────────────────────────────────────────────────────────────\n\njulia> round.(stderror(ols), digits=5)\n2-element Array{Float64,1}:\n 0.62361\n 0.28868\n\njulia> round.(predict(ols), digits=5)\n3-element Array{Float64,1}:\n 1.83333\n 4.33333\n 6.83333"
},

{
    "location": "examples/#Probit-regression-1",
    "page": "Examples",
    "title": "Probit regression",
    "category": "section",
    "text": "julia> data = DataFrame(X=[1,2,2], Y=[1,0,1])\n3×2 DataFrames.DataFrame\n│ Row │ X     │ Y     │\n│     │ Int64 │ Int64 │\n├─────┼───────┼───────┤\n│ 1   │ 1     │ 1     │\n│ 2   │ 2     │ 0     │\n│ 3   │ 2     │ 1     │\n\njulia> probit = glm(@formula(Y ~ X), data, Binomial(), ProbitLink())\nStatsModels.DataFrameRegressionModel{GeneralizedLinearModel{GlmResp{Array{Float64,1},Binomial{Float64},ProbitLink},DensePredChol{Float64,LinearAlgebra.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}\n\nFormula: Y ~ 1 + X\n\nCoefficients:\n────────────────────────────────────────────────────────────────────────\n                Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%\n────────────────────────────────────────────────────────────────────────\n(Intercept)   9.63839     293.909   0.03    0.9738   -566.414    585.69\nX            -4.81919     146.957  -0.03    0.9738   -292.849    283.211\n────────────────────────────────────────────────────────────────────────"
},

{
    "location": "examples/#Negative-binomial-regression-1",
    "page": "Examples",
    "title": "Negative binomial regression",
    "category": "section",
    "text": "julia> using GLM, RDatasets\n\njulia> quine = dataset(\"MASS\", \"quine\")\n146×5 DataFrames.DataFrame\n│ Row │ Eth          │ Sex          │ Age          │ Lrn          │ Days  │\n│     │ Categorical… │ Categorical… │ Categorical… │ Categorical… │ Int32 │\n├─────┼──────────────┼──────────────┼──────────────┼──────────────┼───────┤\n│ 1   │ A            │ M            │ F0           │ SL           │ 2     │\n│ 2   │ A            │ M            │ F0           │ SL           │ 11    │\n│ 3   │ A            │ M            │ F0           │ SL           │ 14    │\n│ 4   │ A            │ M            │ F0           │ AL           │ 5     │\n│ 5   │ A            │ M            │ F0           │ AL           │ 5     │\n│ 6   │ A            │ M            │ F0           │ AL           │ 13    │\n│ 7   │ A            │ M            │ F0           │ AL           │ 20    │\n⋮\n│ 139 │ N            │ F            │ F3           │ AL           │ 22    │\n│ 140 │ N            │ F            │ F3           │ AL           │ 3     │\n│ 141 │ N            │ F            │ F3           │ AL           │ 3     │\n│ 142 │ N            │ F            │ F3           │ AL           │ 5     │\n│ 143 │ N            │ F            │ F3           │ AL           │ 15    │\n│ 144 │ N            │ F            │ F3           │ AL           │ 18    │\n│ 145 │ N            │ F            │ F3           │ AL           │ 22    │\n│ 146 │ N            │ F            │ F3           │ AL           │ 37    │\n\njulia> nbrmodel = glm(@formula(Days ~ Eth+Sex+Age+Lrn), quine, NegativeBinomial(2.0), LogLink())\nStatsModels.DataFrameRegressionModel{GeneralizedLinearModel{GlmResp{Array{Float64,1},NegativeBinomial{Float64},LogLink},DensePredChol{Float64,LinearAlgebra.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}\n\nFormula: Days ~ 1 + Eth + Sex + Age + Lrn\n\nCoefficients:\n────────────────────────────────────────────────────────────────────────────\n                  Coef.  Std. Error      z  Pr(>|z|)   Lower 95%   Upper 95%\n────────────────────────────────────────────────────────────────────────────\n(Intercept)   2.88645      0.227144  12.71    <1e-36   2.44125     3.33164\nEth: N       -0.567515     0.152449  -3.72    0.0002  -0.86631    -0.26872\nSex: M        0.0870771    0.159025   0.55    0.5840  -0.224606    0.398761\nAge: F1      -0.445076     0.239087  -1.86    0.0627  -0.913678    0.0235251\nAge: F2       0.0927999    0.234502   0.40    0.6923  -0.366816    0.552416\nAge: F3       0.359485     0.246586   1.46    0.1449  -0.123814    0.842784\nLrn: SL       0.296768     0.185934   1.60    0.1105  -0.0676559   0.661191\n────────────────────────────────────────────────────────────────────────────\n\njulia> nbrmodel = negbin(@formula(Days ~ Eth+Sex+Age+Lrn), quine, LogLink())\nStatsModels.DataFrameRegressionModel{GeneralizedLinearModel{GlmResp{Array{Float64,1},NegativeBinomial{Float64},LogLink},DensePredChol{Float64,LinearAlgebra.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}\n\nFormula: Days ~ 1 + Eth + Sex + Age + Lrn\n\nCoefficients:\n────────────────────────────────────────────────────────────────────────────\n                  Coef.  Std. Error      z  Pr(>|z|)   Lower 95%   Upper 95%\n────────────────────────────────────────────────────────────────────────────\n(Intercept)   2.89453      0.227415  12.73    <1e-36   2.4488      3.34025\nEth: N       -0.569341     0.152656  -3.73    0.0002  -0.868541   -0.270141\nSex: M        0.0823881    0.159209   0.52    0.6048  -0.229655    0.394431\nAge: F1      -0.448464     0.238687  -1.88    0.0603  -0.916281    0.0193536\nAge: F2       0.0880506    0.235149   0.37    0.7081  -0.372834    0.548935\nAge: F3       0.356955     0.247228   1.44    0.1488  -0.127602    0.841513\nLrn: SL       0.292138     0.18565    1.57    0.1156  -0.0717297   0.656006\n────────────────────────────────────────────────────────────────────────────\n\njulia> println(\"Estimated theta = \", round(nbrmodel.model.rr.d.r, digits=5))\nEstimated theta = 1.27489\n"
},

{
    "location": "examples/#Julia-and-R-comparisons-1",
    "page": "Examples",
    "title": "Julia and R comparisons",
    "category": "section",
    "text": "An example of a simple linear model in R is> coef(summary(lm(optden ~ carb, Formaldehyde)))\n               Estimate  Std. Error    t value     Pr(>|t|)\n(Intercept) 0.005085714 0.007833679  0.6492115 5.515953e-01\ncarb        0.876285714 0.013534536 64.7444207 3.409192e-07The corresponding model with the GLM package isjulia> using GLM, RDatasets\n\njulia> form = dataset(\"datasets\", \"Formaldehyde\")\n6×2 DataFrame\n│ Row │ Carb     │ OptDen   │\n│     │ Float64⍰ │ Float64⍰ │\n├─────┼──────────┼──────────┤\n│ 1   │ 0.1      │ 0.086    │\n│ 2   │ 0.3      │ 0.269    │\n│ 3   │ 0.5      │ 0.446    │\n│ 4   │ 0.6      │ 0.538    │\n│ 5   │ 0.7      │ 0.626    │\n│ 6   │ 0.9      │ 0.782    │\n\njulia> lm1 = fit(LinearModel, @formula(OptDen ~ Carb), form)\nStatsModels.DataFrameRegressionModel{LinearModel{LmResp{Array{Float64,1}},DensePredChol{Float64,LinearAlgebra.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}\n\nFormula: OptDen ~ 1 + Carb\n\n───────────────────────────────────────────────────────────────────────────\n                  Coef.  Std. Error      t  Pr(>|t|)   Lower 95%  Upper 95%\n───────────────────────────────────────────────────────────────────────────\n(Intercept)  0.00508571  0.00783368   0.65    0.5516  -0.0166641  0.0268355\nCarb         0.876286    0.0135345   64.74    <1e-6    0.838708   0.913864\n───────────────────────────────────────────────────────────────────────────A more complex example in R is> coef(summary(lm(sr ~ pop15 + pop75 + dpi + ddpi, LifeCycleSavings)))\n                 Estimate   Std. Error    t value     Pr(>|t|)\n(Intercept) 28.5660865407 7.3545161062  3.8841558 0.0003338249\npop15       -0.4611931471 0.1446422248 -3.1885098 0.0026030189\npop75       -1.6914976767 1.0835989307 -1.5609998 0.1255297940\ndpi         -0.0003369019 0.0009311072 -0.3618293 0.7191731554\nddpi         0.4096949279 0.1961971276  2.0881801 0.0424711387with the corresponding Julia codejulia> LifeCycleSavings = dataset(\"datasets\", \"LifeCycleSavings\")\n50×6 DataFrame\n│ Row │ Country        │ SR       │ Pop15    │ Pop75    │ DPI      │ DDPI     │\n│     │ String⍰        │ Float64⍰ │ Float64⍰ │ Float64⍰ │ Float64⍰ │ Float64⍰ │\n├─────┼────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤\n│ 1   │ Australia      │ 11.43    │ 29.35    │ 2.87     │ 2329.68  │ 2.87     │\n│ 2   │ Austria        │ 12.07    │ 23.32    │ 4.41     │ 1507.99  │ 3.93     │\n│ 3   │ Belgium        │ 13.17    │ 23.8     │ 4.43     │ 2108.47  │ 3.82     │\n│ 4   │ Bolivia        │ 5.75     │ 41.89    │ 1.67     │ 189.13   │ 0.22     │\n│ 5   │ Brazil         │ 12.88    │ 42.19    │ 0.83     │ 728.47   │ 4.56     │\n│ 6   │ Canada         │ 8.79     │ 31.72    │ 2.85     │ 2982.88  │ 2.43     │\n│ 7   │ Chile          │ 0.6      │ 39.74    │ 1.34     │ 662.86   │ 2.67     │\n⋮\n│ 43  │ United Kingdom │ 7.81     │ 23.27    │ 4.46     │ 1813.93  │ 2.01     │\n│ 44  │ United States  │ 7.56     │ 29.81    │ 3.43     │ 4001.89  │ 2.45     │\n│ 45  │ Venezuela      │ 9.22     │ 46.4     │ 0.9      │ 813.39   │ 0.53     │\n│ 46  │ Zambia         │ 18.56    │ 45.25    │ 0.56     │ 138.33   │ 5.14     │\n│ 47  │ Jamaica        │ 7.72     │ 41.12    │ 1.73     │ 380.47   │ 10.23    │\n│ 48  │ Uruguay        │ 9.24     │ 28.13    │ 2.72     │ 766.54   │ 1.88     │\n│ 49  │ Libya          │ 8.89     │ 43.69    │ 2.07     │ 123.58   │ 16.71    │\n│ 50  │ Malaysia       │ 4.71     │ 47.2     │ 0.66     │ 242.69   │ 5.08     │\n\njulia> fm2 = fit(LinearModel, @formula(SR ~ Pop15 + Pop75 + DPI + DDPI), LifeCycleSavings)\nStatsModels.DataFrameRegressionModel{LinearModel{LmResp{Array{Float64,1}},DensePredChol{Float64,LinearAlgebra.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}\n\nFormula: SR ~ 1 + Pop15 + Pop75 + DPI + DDPI\n\nCoefficients:\n─────────────────────────────────────────────────────────────────────────────────\n                    Coef.   Std. Error      t  Pr(>|t|)    Lower 95%    Upper 95%\n─────────────────────────────────────────────────────────────────────────────────\n(Intercept)  28.5661       7.35452       3.88    0.0003  13.7533      43.3788\nPop15        -0.461193     0.144642     -3.19    0.0026  -0.752518    -0.169869\nPop75        -1.6915       1.0836       -1.56    0.1255  -3.87398      0.490983\nDPI          -0.000336902  0.000931107  -0.36    0.7192  -0.00221225   0.00153844\nDDPI          0.409695     0.196197      2.09    0.0425   0.0145336    0.804856\n─────────────────────────────────────────────────────────────────────────────────The glm function (or equivalently, fit(GeneralizedLinearModel, ...)) works similarly to the R glm function except that the family argument is replaced by a Distribution type and, optionally, a Link type. The first example from ?glm in R isglm> ## Dobson (1990) Page 93: Randomized Controlled Trial : (slightly modified)\nglm> counts <- c(18,17,15,20,10,21,25,13,13)\n\nglm> outcome <- gl(3,1,9)\n\nglm> treatment <- gl(3,3)\n\nglm> print(d.AD <- data.frame(treatment, outcome, counts))\n  treatment outcome counts\n1         1       1     18\n2         1       2     17\n3         1       3     15\n4         2       1     20\n5         2       2     10\n6         2       3     21\n7         3       1     25\n8         3       2     13\n9         3       3     13\n\nglm> glm.D93 <- glm(counts ~ outcome + treatment, family=poisson())\n\nglm> anova(glm.D93)\nAnalysis of Deviance Table\n\nModel: poisson, link: log\n\nResponse: counts\n\nTerms added sequentially (first to last)\n\n\n          Df Deviance Resid. Df Resid. Dev\nNULL                          8    10.3928\noutcome    2   5.2622         6     5.1307\ntreatment  2   0.0132         4     5.1175\n\nglm> ## No test:\nglm> summary(glm.D93)\n\nCall:\nglm(formula = counts ~ outcome + treatment, family = poisson())\n\nDeviance Residuals: \n      1        2        3        4        5        6        7        8        9  \n-0.6122   1.0131  -0.2819  -0.2498  -0.9784   1.0777   0.8162  -0.1155  -0.8811  \n\nCoefficients:\n            Estimate Std. Error z value Pr(>|z|)    \n(Intercept)   3.0313     0.1712  17.711   <2e-16 ***\noutcome2     -0.4543     0.2022  -2.247   0.0246 *  \noutcome3     -0.2513     0.1905  -1.319   0.1870    \ntreatment2    0.0198     0.1990   0.100   0.9207    \ntreatment3    0.0198     0.1990   0.100   0.9207    \n---\nSignif. codes:  0 \'***\' 0.001 \'**\' 0.01 \'*\' 0.05 \'.\' 0.1 \' \' 1\n\n(Dispersion parameter for poisson family taken to be 1)\n\n    Null deviance: 10.3928  on 8  degrees of freedom\nResidual deviance:  5.1175  on 4  degrees of freedom\nAIC: 56.877\n\nNumber of Fisher Scoring iterations: 4In Julia this becomesjulia> using DataFrames, CategoricalArrays, GLM\n\njulia> dobson = DataFrame(Counts    = [18.,17,15,20,10,21,25,13,13],\n                          Outcome   = categorical([1,2,3,1,2,3,1,2,3]),\n                          Treatment = categorical([1,1,1,2,2,2,3,3,3]))\n9×3 DataFrame\n│ Row │ Counts  │ Outcome      │ Treatment    │\n│     │ Float64 │ Categorical… │ Categorical… │\n├─────┼─────────┼──────────────┼──────────────┤\n│ 1   │ 18.0    │ 1            │ 1            │\n│ 2   │ 17.0    │ 2            │ 1            │\n│ 3   │ 15.0    │ 3            │ 1            │\n│ 4   │ 20.0    │ 1            │ 2            │\n│ 5   │ 10.0    │ 2            │ 2            │\n│ 6   │ 21.0    │ 3            │ 2            │\n│ 7   │ 25.0    │ 1            │ 3            │\n│ 8   │ 13.0    │ 2            │ 3            │\n│ 9   │ 13.0    │ 3            │ 3            │\n\n\njulia> gm1 = fit(GeneralizedLinearModel, @formula(Counts ~ Outcome + Treatment), dobson, Poisson())\nStatsModels.DataFrameRegressionModel{GeneralizedLinearModel{GlmResp{Array{Float64,1},Poisson{Float64},LogLink},DensePredChol{Float64,LinearAlgebra.Cholesky{Float64,Array{Float64,2}}}},Array{Float64,2}}\n\nFormula: Counts ~ 1 + Outcome + Treatment\n\nCoefficients:\n────────────────────────────────────────────────────────────────────────────\n                   Coef.  Std. Error      z  Pr(>|z|)  Lower 95%   Upper 95%\n────────────────────────────────────────────────────────────────────────────\n(Intercept)    3.03128      0.171155  17.71    <1e-69   2.69582    3.36674\nOutcome: 2    -0.454255     0.202171  -2.25    0.0246  -0.850503  -0.0580079\nOutcome: 3    -0.251314     0.190476  -1.32    0.1870  -0.624641   0.122012\nTreatment: 2   0.0198026    0.199017   0.10    0.9207  -0.370264   0.409869\nTreatment: 3   0.0198026    0.199017   0.10    0.9207  -0.370264   0.409869\n────────────────────────────────────────────────────────────────────────────\n\njulia> round(deviance(gm1), digits=5)\n5.11746"
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
    "location": "api/#Types-defined-in-the-package-1",
    "page": "API",
    "title": "Types defined in the package",
    "category": "section",
    "text": "DensePredChol\nDensePredQR\nGlmResp\nLinearModel\nLmResp\nLinPred\nGLM.ModResp"
},

{
    "location": "api/#GLM.glm",
    "page": "API",
    "title": "GLM.glm",
    "category": "function",
    "text": "glm(F, D, args...; kwargs...)\n\nFit a generalized linear model to data. Alias for fit(GeneralizedLinearModel, ...). See fit for documentation.\n\n\n\n\n\n"
},

{
    "location": "api/#StatsBase.fit",
    "page": "API",
    "title": "StatsBase.fit",
    "category": "function",
    "text": "fit(GeneralizedLinearModel, X, y, d, [l = canonicallink(d)]; <keyword arguments>)\n\nFit a generalized linear model to data. X and y can either be a matrix and a vector, respectively, or a formula and a data frame. d must be a UnivariateDistribution, and l must be a Link, if supplied.\n\nKeyword Arguments\n\ndofit::Bool=true: Determines whether model will be fit\nwts::Vector=similar(y,0): Prior frequency (a.k.a. case) weights of observations.\n\nSuch weights are equivalent to repeating each observation a number of times equal to its weight. Do note that this interpretation gives equal point estimates but different standard errors from analytical (a.k.a. inverse variance) weights and from probability (a.k.a. sampling) weights which are the default in some other software. Can be length 0 to indicate no weighting (default).\n\noffset::Vector=similar(y,0): offset added to Xβ to form eta.  Can be of\n\nlength 0\n\nverbose::Bool=false: Display convergence information for each iteration\nmaxiter::Integer=30: Maximum number of iterations allowed to achieve convergence\natol::Real=1e-6: Convergence is achieved when the relative change in\n\ndeviance is less than max(rtol*dev, atol).\n\nrtol::Real=1e-6: Convergence is achieved when the relative change in\n\ndeviance is less than max(rtol*dev, atol).\n\nminstepfac::Real=0.001: Minimum line step fraction. Must be between 0 and 1.\nstart::AbstractVector=nothing: Starting values for beta. Should have the\n\nsame length as the number of columns in the model matrix.\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.lm",
    "page": "API",
    "title": "GLM.lm",
    "category": "function",
    "text": "lm(X, y, allowrankdeficient::Bool=false; wts=similar(y, 0))\n\nAn alias for fit(LinearModel, X, y, allowrankdeficient)\n\nThe arguments X and y can be a Matrix and a Vector or a Formula and a DataFrame.\n\nThe keyword argument wts can be a Vector specifying frequency weights for observations. Such weights are equivalent to repeating each observation a number of times equal to its weight. Do note that this interpretation gives equal point estimates but different standard errors from analytical (a.k.a. inverse variance) weights and from probability (a.k.a. sampling) weights which are the default in some other software.\n\n\n\n\n\n"
},

{
    "location": "api/#GLM.negbin",
    "page": "API",
    "title": "GLM.negbin",
    "category": "function",
    "text": "negbin(formula,\n       data,\n       link;\n       initialθ::Real=Inf,\n       maxiter::Integer=30,\n       atol::Real=1e-6,\n       rtol::Real=1.e-6,\n       verbose::Bool=false,\n       kwargs...)\n\nFit a negative binomial generalized linear model to data, while simultaneously estimating the shape parameter θ. Extra arguments and keyword arguments will be passed to glm.\n\nKeyword Arguments\n\ninitialθ::Real=Inf: Starting value for shape parameter θ. If it is Inf then the initial value will be estimated by fitting a Poisson distribution.\nmaxiter::Integer=30: See maxiter for glm\natol::Real=1.0e-6: See atol for glm\nrtol::Real=1.0e-6: See rtol for glm\nverbose::Bool=false: See verbose for glm\n\n\n\n\n\n"
},

{
    "location": "api/#Constructors-for-models-1",
    "page": "API",
    "title": "Constructors for models",
    "category": "section",
    "text": "The most general approach to fitting a model is with the fit function, as injulia> using Random\n\njulia> fit(LinearModel, hcat(ones(10), 1:10), randn(MersenneTwister(12321), 10))\nLinearModel{GLM.LmResp{Array{Float64,1}},GLM.DensePredChol{Float64,LinearAlgebra.Cholesky{Float64,Array{Float64,2}}}}:\n\nCoefficients:\n────────────────────────────────────────────────────────────────\n        Coef.  Std. Error      t  Pr(>|t|)  Lower 95%  Upper 95%\n────────────────────────────────────────────────────────────────\nx1   0.717436    0.775175   0.93    0.3818  -1.07012    2.50499\nx2  -0.152062    0.124931  -1.22    0.2582  -0.440153   0.136029\n────────────────────────────────────────────────────────────────This model can also be fit asjulia> using Random\n\njulia> lm(hcat(ones(10), 1:10), randn(MersenneTwister(12321), 10))\nLinearModel{LmResp{Array{Float64,1}},DensePredChol{Float64,LinearAlgebra.Cholesky{Float64,Array{Float64,2}}}}:\n\nCoefficients:\n────────────────────────────────────────────────────────────────\n        Coef.  Std. Error      t  Pr(>|t|)  Lower 95%  Upper 95%\n────────────────────────────────────────────────────────────────\nx1   0.717436    0.775175   0.93    0.3818  -1.07012    2.50499\nx2  -0.152062    0.124931  -1.22    0.2582  -0.440153   0.136029\n────────────────────────────────────────────────────────────────glm\nfit\nlm\nnegbin"
},

{
    "location": "api/#Model-methods-1",
    "page": "API",
    "title": "Model methods",
    "category": "section",
    "text": "GLM.cancancel\ndelbeta!\nStatsBase.deviance\nGLM.dispersion\nGLM.installbeta!\nGLM.issubmodel\nlinpred!\nlinpred\nStatsBase.nobs\nStatsBase.nulldeviance\nStatsBase.predict\nupdateμ!\nwrkresp\nGLM.wrkresp!"
},

{
    "location": "api/#Links-and-methods-applied-to-them-1",
    "page": "API",
    "title": "Links and methods applied to them",
    "category": "section",
    "text": "Link\nGLM.Link01\nCauchitLink\nCloglogLink\nIdentityLink\nInverseLink\nInverseSquareLink\nLogitLink\nLogLink\nNegativeBinomialLink\nProbitLink\nSqrtLink\nlinkfun\nlinkinv\nmueta\ninverselink\ncanonicallink\nglmvar\nmustart\ndevresid\nGLM.dispersion_parameter\nGLM.loglik_obs"
},

]}
