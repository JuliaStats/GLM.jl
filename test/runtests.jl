using Base.Test, DataFrames, GLM

function test_show(x)
	io = IOBuffer()
	show(io, x)
end

## Formaldehyde data from the R Datasets package
form = DataFrame(Carb=[0.1,0.3,0.5,0.6,0.7,0.9],OptDen=[0.086,0.269,0.446,0.538,0.626,0.782])
lm1 = fit(LinearModel, OptDen ~ Carb, form)
test_show(lm1)
@test_approx_eq coef(lm1) linreg(convert(Array, form[:Carb]), convert(Array, form[:OptDen]))
Σ = [6.136653061224592e-05 -9.464489795918525e-05
    -9.464489795918525e-05 1.831836734693908e-04]
@test_approx_eq vcov(lm1) Σ
@test_approx_eq cor(lm1.model) diagm(diag(Σ))^(-1/2)*Σ*diagm(diag(Σ))^(-1/2)

dobson = DataFrame(Counts=[18.,17,15,20,10,20,25,13,12], Outcome=gl(3,1,9), Treatment=gl(3,3))
gm1 = fit(GeneralizedLinearModel, Counts ~ Outcome + Treatment, dobson, Poisson())
test_show(gm1)
@test_approx_eq deviance(gm1) 5.12914107700115
@test_approx_eq coef(gm1)[1:3] [3.044522437723423,-0.45425527227759555,-0.29298712468147375]

## Example from http://www.ats.ucla.edu/stat/r/dae/logit.htm
df = readtable(Pkg.dir("GLM","data","admit.csv.gz"))
df[:rank] = pool(df[:rank])

gm2 = fit(GeneralizedLinearModel, admit ~ gre + gpa + rank, df, Binomial())
test_show(gm2)
@test_approx_eq deviance(gm2) 458.5174924758994
@test_approx_eq coef(gm2) [-3.9899786606380734,0.0022644256521549043,0.8040374535155766,-0.6754428594116577,-1.3402038117481079,-1.5514636444657492]

gm3 = fit(GeneralizedLinearModel, admit ~ gre + gpa + rank, df, Binomial(), ProbitLink())
test_show(gm3)
@test_approx_eq deviance(gm3) 458.4131713833386
@test_approx_eq coef(gm3) [-2.3867922998680786,0.0013755394922972369,0.47772908362647015,-0.4154125854823675,-0.8121458010130356,-0.9359047862425298]

gm4 = fit(GeneralizedLinearModel, admit ~ gre + gpa + rank, df, Binomial(), CauchitLink())
test_show(gm4)
@test_approx_eq deviance(gm4) 459.3401112751141

gm5 = fit(GeneralizedLinearModel, admit ~ gre + gpa + rank, df, Binomial(), CloglogLink())
test_show(gm5)
@test_approx_eq deviance(gm5) 458.89439629612616

## Example with offsets from Venables & Ripley (2002, p.189)
df = readtable(Pkg.dir("GLM","data","anorexia.csv.gz"))
df[:Treat] = pool(df[:Treat])

gm6 = fit(GeneralizedLinearModel, Postwt ~ Prewt + Treat, df, Normal(), IdentityLink(), offset=convert(Array, df[:Prewt]))
test_show(gm6)
@test_approx_eq deviance(gm6) 3311.262619919613
@test_approx_eq coef(gm6) [49.7711090149846,-0.5655388496391,-4.0970655280729,4.5630626529188]
@test_approx_eq scale(gm6.model, true) 48.6950385282296
@test_approx_eq stderr(gm6) [13.3909581420259,0.1611823618518,1.8934926069669,2.1333359226431]

gm7 = fit(GeneralizedLinearModel, Postwt ~ Prewt + Treat, df, Normal(), LogLink(), offset=convert(Array, df[:Prewt]),
	      convTol=1e-8)
test_show(gm7)
@test_approx_eq deviance(gm7) 3265.207242977156
@test_approx_eq coef(gm7) [3.992326787835955,-0.994452693131178,-0.050698258703974,0.051494029957641]
@test_approx_eq scale(gm7.model, true) 48.01787789178518
@test_approx_eq stderr(gm7) [0.157167944259695,0.001886285986164,0.022584069426311,0.023882826190166]

## Gamma example from McCullagh & Nelder (1989, pp. 300-2)
clotting = DataFrame(u = log([5,10,15,20,30,40,60,80,100]),
                     lot1 = [118,58,42,35,27,25,21,19,18])
gm8 = fit(GeneralizedLinearModel, lot1 ~ u, clotting, Gamma())
test_show(gm8)
@test_approx_eq deviance(gm8) 0.01672971517848353
@test_approx_eq coef(gm8) [-0.01655438172784895,0.01534311491072141]
@test_approx_eq scale(gm8.model, true) 0.002446059333495581
@test_approx_eq stderr(gm8) [0.0009275466067257,0.0004149596425600]

gm9 = fit(GeneralizedLinearModel, lot1 ~ u, clotting, Gamma(), LogLink(), convTol=1e-8)
test_show(gm9)
@test_approx_eq deviance(gm9) 0.16260829451739
@test_approx_eq coef(gm9) [5.50322528458221,-0.60191617825971]
@test_approx_eq scale(gm9.model, true) 0.02435442293561081
@test_approx_eq stderr(gm9) [0.19030107482720,0.05530784660144]

gm10 = fit(GeneralizedLinearModel, lot1 ~ u, clotting, Gamma(), IdentityLink(), convTol=1e-8)
test_show(gm10)
@test_approx_eq deviance(gm10) 0.60845414895344
@test_approx_eq coef(gm10) [99.250446880986,-18.374324929002]
@test_approx_eq scale(gm10.model, true) 0.1041772704067886
@test_approx_eq stderr(gm10) [17.864388462865,4.297968703823]

## Fitting GLMs with sparse matrices
srand(1)
X = sprand(1000, 10, 0.01)
β = randn(10)
y = Bool[rand() < x for x in logistic(X * β)]

gmsparse = fit(GeneralizedLinearModel, X, y, Binomial())
gmdense = fit(GeneralizedLinearModel, full(X), y, Binomial())

@test_approx_eq deviance(gmsparse) deviance(gmdense)
@test_approx_eq coef(gmsparse) coef(gmdense)
@test_approx_eq vcov(gmsparse) vcov(gmdense)

## Prediction for GLMs
srand(1)
X = rand(10, 2)
Y = logistic(X * [3; -3])

gm11 = fit(GeneralizedLinearModel, X, Y, Binomial())
@test_approx_eq predict(gm11) Y

newX = rand(5, 2)
newY = logistic(newX * coef(gm11))
@test_approx_eq predict(gm11, newX) newY

off = rand(10)
newoff = rand(5)

@test_throws ArgumentError predict(gm11, newX, offset=newoff)

gm12 = fit(GeneralizedLinearModel, X, Y, Binomial(), offset=off)
@test_throws ArgumentError predict(gm12, newX)
@test_approx_eq predict(gm12, newX, offset=newoff) logistic(newX * coef(gm12) .+ newoff)

## Prediction from DataFrames
d = convert(DataFrame, X)
d[:y] = Y

gm13 = fit(GeneralizedLinearModel, y ~ 0 + x1 + x2, d, Binomial())
@test predict(gm13) == predict(gm13, d[[:x1, :x2]])
@test predict(gm13) == predict(gm13, d)

newd = convert(DataFrame, newX)
predict(gm13, newd)
