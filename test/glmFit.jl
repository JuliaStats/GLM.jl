using Base.Test, GLM, DataFrames

## Formaldehyde data from the R Datasets package
form = DataFrame(Carb=[0.1,0.3,0.5,0.6,0.7,0.9],OptDen=[0.086,0.269,0.446,0.538,0.626,0.782])
lm1 = fit(LinearModel, OptDen ~ Carb, form)
@test_approx_eq coef(lm1) linreg(array(form[:Carb]),array(form[:OptDen]))

dobson = DataFrame(Counts=[18.,17,15,20,10,20,25,13,12], Outcome=gl(3,1,9), Treatment=gl(3,3))
gm1 = fit(GeneralizedLinearModel, Counts ~ Outcome + Treatment, dobson, Poisson());
@test_approx_eq deviance(gm1) 5.12914107700115
@test_approx_eq coef(gm1)[1:3] [3.044522437723423,-0.45425527227759555,-0.29298712468147375]

## Example from http://www.ats.ucla.edu/stat/r/dae/logit.htm
df = readtable(Pkg.dir("GLM","data","admit.csv.gz"))
df[:rank] = pool(df[:rank])

gm2 = fit(GeneralizedLinearModel, admit ~ gre + gpa + rank, df, Binomial())
@test_approx_eq deviance(gm2) 458.5174924758994
@test_approx_eq coef(gm2) [-3.9899786606380734,0.0022644256521549043,0.8040374535155766,-0.6754428594116577,-1.3402038117481079,-1.5514636444657492]

gm3 = fit(GeneralizedLinearModel, admit ~ gre + gpa + rank, df, Binomial(), ProbitLink())
@test_approx_eq deviance(gm3) 458.4131713833386
@test_approx_eq coef(gm3) [-2.3867922998680786,0.0013755394922972369,0.47772908362647015,-0.4154125854823675,-0.8121458010130356,-0.9359047862425298]

gm4 = fit(GeneralizedLinearModel, admit ~ gre + gpa + rank, df, Binomial(), CauchitLink())
@test_approx_eq deviance(gm4) 459.3401112751141

gm5 = fit(GeneralizedLinearModel, admit ~ gre + gpa + rank, df, Binomial(), CloglogLink())
@test_approx_eq deviance(gm5) 458.89439629612616

## Example with offsets from Venables & Ripley (2002, p.189)
df = readtable(Pkg.dir("GLM","data","anorexia.csv.gz"))
df[:Treat] = pool(df[:Treat])

gm6 = fit(GeneralizedLinearModel, Postwt ~ Prewt + Treat, df, Normal(), IdentityLink(), offset=array(df[:Prewt]))
@test_approx_eq deviance(gm6) 3311.262619919613
@test_approx_eq coef(gm6) [49.7711090149846,-0.5655388496391,-4.0970655280729,4.5630626529188]
@test_approx_eq scale(gm6.model, true) 48.6950385282296
@test_approx_eq stderr(gm6) [13.3909581420259,0.1611823618518,1.8934926069669,2.1333359226431]

gm7 = fit(GeneralizedLinearModel, Postwt ~ Prewt + Treat, df, Normal(), LogLink(), offset=array(df[:Prewt]),
	      convTol=1e-8)
@test_approx_eq deviance(gm7) 3265.207242977156
@test_approx_eq coef(gm7) [3.992326787835955,-0.994452693131178,-0.050698258703974,0.051494029957641]
@test_approx_eq scale(gm7.model, true) 48.01787789178518
@test_approx_eq stderr(gm7) [0.157167944259695,0.001886285986164,0.022584069426311,0.023882826190166]

## Gamma example from McCullagh & Nelder (1989, pp. 300-2)
clotting = DataFrame(u = log([5,10,15,20,30,40,60,80,100]),
                     lot1 = [118,58,42,35,27,25,21,19,18])
gm8 = fit(GeneralizedLinearModel, lot1 ~ u, clotting, Gamma())
@test_approx_eq deviance(gm8) 0.01672971517848353
@test_approx_eq coef(gm8) [-0.01655438172784895,0.01534311491072141]
@test_approx_eq scale(gm8.model, true) 0.002446059333495581
@test_approx_eq stderr(gm8) [0.0009275466067257,0.0004149596425600]

gm9 = fit(GeneralizedLinearModel, lot1 ~ u, clotting, Gamma(), LogLink(), convTol=1e-8)
@test_approx_eq deviance(gm9) 0.16260829451739
@test_approx_eq coef(gm9) [5.50322528458221,-0.60191617825971]
@test_approx_eq scale(gm9.model, true) 0.02435442293561081
@test_approx_eq stderr(gm9) [0.19030107482720,0.05530784660144]

gm10 = fit(GeneralizedLinearModel, lot1 ~ u, clotting, Gamma(), IdentityLink(), convTol=1e-8)
@test_approx_eq deviance(gm10) 0.60845414895344
@test_approx_eq coef(gm10) [99.250446880986,-18.374324929002]
@test_approx_eq scale(gm10.model, true) 0.1041772704067886
@test_approx_eq stderr(gm10) [17.864388462865,4.297968703823]
