using Base.Test, StatsFuns, DataFrames, GLM

function test_show(x)
    io = IOBuffer()
    show(io, x)
end

const glm_datadir = joinpath(dirname(@__FILE__), "..", "data")

## Formaldehyde data from the R Datasets package
form = DataFrame(Carb=[0.1,0.3,0.5,0.6,0.7,0.9],OptDen=[0.086,0.269,0.446,0.538,0.626,0.782])
lm1 = fit(LinearModel, OptDen ~ Carb, form)
test_show(lm1)
@test_approx_eq coef(lm1) collect(linreg(convert(Array, form[:Carb]), convert(Array, form[:OptDen])))
Σ = [6.136653061224592e-05 -9.464489795918525e-05
    -9.464489795918525e-05 1.831836734693908e-04]
@test_approx_eq vcov(lm1) Σ
@test_approx_eq cor(lm1.model) diagm(diag(Σ))^(-1/2)*Σ*diagm(diag(Σ))^(-1/2)
@test df(lm1) == 3
@test_approx_eq deviance(lm1) 0.0002992000000000012
@test_approx_eq loglikelihood(lm1) 21.204842144047973
@test_approx_eq nulldeviance(lm1) 0.3138488333333334
@test_approx_eq nullloglikelihood(lm1) 0.33817870295676444
@test r²(lm1) == r2(lm1)
@test_approx_eq r²(lm1) 0.9990466748057584
@test adjr²(lm1) == adjr2(lm1)
@test_approx_eq adjr²(lm1) 0.998808343507198
@test_approx_eq aic(lm1) -36.409684288095946
@test_approx_eq aicc(lm1) -24.409684288095946
@test_approx_eq bic(lm1) -37.03440588041178

dobson = DataFrame(Counts=[18.,17,15,20,10,20,25,13,12], Outcome=gl(3,1,9), Treatment=gl(3,3))
gm1 = fit(GeneralizedLinearModel, Counts ~ Outcome + Treatment, dobson, Poisson())
test_show(gm1)
@test df(gm1) == 5
@test_approx_eq deviance(gm1) 5.12914107700115
@test_approx_eq loglikelihood(gm1) -23.380659200978837
@test_approx_eq aic(gm1) 56.76131840195767
@test_approx_eq aicc(gm1) 76.76131840195768
@test_approx_eq bic(gm1) 57.74744128863877
@test_approx_eq coef(gm1)[1:3] [3.044522437723423,-0.45425527227759555,-0.29298712468147375]

## Example from http://www.ats.ucla.edu/stat/r/dae/logit.htm
admit = readtable(joinpath(glm_datadir, "admit.csv.gz"))
admit[:rank] = pool(admit[:rank])

for distr in (Binomial, Bernoulli)
    gm2 = fit(GeneralizedLinearModel, admit ~ gre + gpa + rank, admit, distr())
    test_show(gm2)
    @test df(gm2) == 6
    @test_approx_eq deviance(gm2) 458.5174924758994
    @test_approx_eq loglikelihood(gm2) -229.25874623794968
    @test_approx_eq aic(gm2) 470.51749247589936
    @test_approx_eq aicc(gm2) 470.7312329339146
    @test_approx_eq bic(gm2) 494.4662797585473
    @test_approx_eq coef(gm2) [-3.9899786606380734,0.0022644256521549043,0.8040374535155766,-0.6754428594116577,-1.3402038117481079,-1.5514636444657492]
end

gm3 = fit(GeneralizedLinearModel, admit ~ gre + gpa + rank, admit, Binomial(), ProbitLink())
test_show(gm3)
@test df(gm3) == 6
@test_approx_eq deviance(gm3) 458.4131713833386
@test_approx_eq loglikelihood(gm3) -229.20658569166932
@test_approx_eq aic(gm3) 470.41317138333864
@test_approx_eq aicc(gm3) 470.6269118413539
@test_approx_eq bic(gm3) 494.36195866598655
@test_approx_eq coef(gm3) [-2.3867922998680786,0.0013755394922972369,0.47772908362647015,-0.4154125854823675,-0.8121458010130356,-0.9359047862425298]

gm4 = fit(GeneralizedLinearModel, admit ~ gre + gpa + rank, admit, Binomial(), CauchitLink())
test_show(gm4)
@test df(gm4) == 6
@test_approx_eq deviance(gm4) 459.3401112751141
@test_approx_eq loglikelihood(gm4) -229.6700556375571
@test_approx_eq aic(gm4) 471.3401112751142
@test_approx_eq aicc(gm4) 471.5538517331295
@test_approx_eq bic(gm4) 495.28889855776214

gm5 = fit(GeneralizedLinearModel, admit ~ gre + gpa + rank, admit, Binomial(), CloglogLink())
test_show(gm5)
@test df(gm5) == 6
@test_approx_eq deviance(gm5) 458.89439629612616
@test_approx_eq loglikelihood(gm5) -229.44719814806314
@test_approx_eq aic(gm5) 470.8943962961263
@test_approx_eq aicc(gm5) 471.1081367541415
@test_approx_eq bic(gm5) 494.8431835787742

## Example with offsets from Venables & Ripley (2002, p.189)
anorexia = readtable(joinpath(glm_datadir, "anorexia.csv.gz"))
anorexia[:Treat] = pool(anorexia[:Treat])

gm6 = fit(GeneralizedLinearModel, Postwt ~ Prewt + Treat, anorexia, Normal(), IdentityLink(), offset=convert(Array, anorexia[:Prewt]))
test_show(gm6)
@test df(gm6) == 5
@test_approx_eq deviance(gm6) 3311.262619919613
@test_approx_eq loglikelihood(gm6) -239.9866487711122
@test_approx_eq aic(gm6) 489.9732975422244
@test_approx_eq aicc(gm6) 490.8823884513153
@test_approx_eq bic(gm6) 501.35662813730465
@test_approx_eq coef(gm6) [49.7711090149846,-0.5655388496391,-4.0970655280729,4.5630626529188]
@test_approx_eq GLM.dispersion(gm6.model, true) 48.6950385282296
@test_approx_eq stderr(gm6) [13.3909581420259,0.1611823618518,1.8934926069669,2.1333359226431]

gm7 = fit(GeneralizedLinearModel, Postwt ~ Prewt + Treat, anorexia, Normal(), LogLink(), offset=convert(Array, anorexia[:Prewt]),
          convTol=1e-8)
test_show(gm7)
@test_approx_eq deviance(gm7) 3265.207242977156
@test_approx_eq coef(gm7) [3.992326787835955,-0.994452693131178,-0.050698258703974,0.051494029957641]
@test_approx_eq GLM.dispersion(gm7.model, true) 48.01787789178518
@test_approx_eq stderr(gm7) [0.157167944259695,0.001886285986164,0.022584069426311,0.023882826190166]

## Gamma example from McCullagh & Nelder (1989, pp. 300-2)
clotting = DataFrame(u = log.([5,10,15,20,30,40,60,80,100]),
                     lot1 = [118,58,42,35,27,25,21,19,18])
gm8 = fit(GeneralizedLinearModel, lot1 ~ u, clotting, Gamma())
test_show(gm8)
@test df(gm8) == 3
@test_approx_eq deviance(gm8) 0.016729715178484157
@test_approx_eq loglikelihood(gm8) -15.994961974777247
@test_approx_eq aic(gm8) 37.989923949554495
@test_approx_eq aicc(gm8) 42.78992394955449
@test_approx_eq bic(gm8) 38.58159768156315
@test_approx_eq coef(gm8) [-0.01655438172784895,0.01534311491072141]
@test_approx_eq GLM.dispersion(gm8.model, true) 0.002446059333495581
@test_approx_eq stderr(gm8) [0.0009275466067257,0.0004149596425600]

gm9 = fit(GeneralizedLinearModel, lot1 ~ u, clotting, Gamma(), LogLink(), convTol=1e-8)
test_show(gm9)
@test df(gm9) == 3
@test_approx_eq deviance(gm9) 0.16260829451739
@test_approx_eq loglikelihood(gm9) -26.24082810384911
@test_approx_eq aic(gm9) 58.48165620769822
@test_approx_eq aicc(gm9) 63.28165620769822
@test_approx_eq bic(gm9) 59.07332993970688
@test_approx_eq coef(gm9) [5.50322528458221,-0.60191617825971]
@test_approx_eq GLM.dispersion(gm9.model, true) 0.02435442293561081
@test_approx_eq stderr(gm9) [0.19030107482720,0.05530784660144]

gm10 = fit(GeneralizedLinearModel, lot1 ~ u, clotting, Gamma(), IdentityLink(), convTol=1e-8)
test_show(gm10)
@test df(gm10) == 3
@test_approx_eq deviance(gm10) 0.60845414895344
@test_approx_eq loglikelihood(gm10) -32.216072437284176
@test_approx_eq aic(gm10) 70.43214487456835
@test_approx_eq aicc(gm10) 75.23214487456835
@test_approx_eq bic(gm10) 71.02381860657701
@test_approx_eq coef(gm10) [99.250446880986,-18.374324929002]
@test_approx_eq GLM.dispersion(gm10.model, true) 0.1041772704067886
@test_approx_eq stderr(gm10) [17.864388462865,4.297968703823]

# Logistic regression using aggregated data and weights
admit_agr = DataFrame(count=[28, 97, 93, 55, 33, 54, 28, 12],
                      admit=repeat([false, true], inner=[4]),
                      rank=PooledDataArray(repeat([1, 2, 3, 4], outer=[2])))
for distr in (Binomial, Bernoulli)
    gm14 = fit(GeneralizedLinearModel, admit ~ rank, admit_agr, distr(), wts=Vector{Float64}(admit_agr[:count]))
    @test df(gm14) == 4
    @test nobs(gm14) == 400
    @test_approx_eq deviance(gm14) 474.9667184280627
    @test_approx_eq loglikelihood(gm14) -237.48335921403134
    @test_approx_eq aic(gm14) 482.96671842822883
    @test_approx_eq aicc(gm14) 483.0679842510136
    @test_approx_eq bic(gm14) 498.9325766164946
    @test_approx_eq_eps coef(gm14) [0.16430305129127593,-0.7500299832244263,-1.364697929944679,-1.6867286645732025] 1e-5
end

# Logistic regression using aggregated data with proportions of successes and weights
admit_agr2 = DataFrame(count=[61, 151, 121, 67],
                       admit=[33, 54, 28, 12],
                       rank=PooledDataArray([1, 2, 3, 4]))
admit_agr2[:p] = admit_agr2[:admit]./admit_agr2[:count]

gm15 = fit(GeneralizedLinearModel, p ~ rank, admit_agr2, Binomial(), wts=Vector{Float64}(admit_agr2[:count]))
test_show(gm15)
@test df(gm15) == 4
@test nobs(gm15) == 400
@test_approx_eq deviance(gm15) -2.4424906541753456e-15
@test_approx_eq loglikelihood(gm15) -9.50254433604239
@test_approx_eq aic(gm15) 27.00508867208478
@test_approx_eq aicc(gm15) 27.106354494869592
@test_approx_eq bic(gm15) 42.970946860516705
@test_approx_eq coef(gm15) [0.1643030512912767,-0.7500299832303851,-1.3646980342693287,-1.6867295867357475]

# Weighted Gamma example (weights are totally made up)
gm16 = fit(GeneralizedLinearModel, lot1 ~ u, clotting, Gamma(), wts=[1.5,2.0,1.1,4.5,2.4,3.5,5.6,5.4,6.7])
test_show(gm16)
@test df(gm16) == 3
@test nobs(gm16) == 32.7
@test_approx_eq deviance(gm16) 0.03933389380881689
@test_approx_eq loglikelihood(gm16) -43.35907878769152
@test_approx_eq aic(gm16) 92.71815757538305
@test_approx_eq aicc(gm16) 93.55439450918095
@test_approx_eq bic(gm16) 97.18028280909267
@test_approx_eq coef(gm16) [-0.017217012615523237,0.015649040411276433]

# Weighted Poisson example (weights are totally made up)
gm17 = fit(GeneralizedLinearModel, Counts ~ Outcome + Treatment, dobson, Poisson(),
           wts=[1.5,2.0,1.1,4.5,2.4,3.5,5.6,5.4,6.7])
test_show(gm17)
@test df(gm17) == 5
@test_approx_eq deviance(gm17) 17.699857821414266
@test_approx_eq loglikelihood(gm17) -84.57429468506352
@test_approx_eq aic(gm17) 179.14858937012704
@test_approx_eq aicc(gm17) 181.39578038136298
@test_approx_eq bic(gm17) 186.5854647596431
@test_approx_eq coef(gm17) [3.1218557035404793,  -0.5270435906931427,-0.40300384148562746,
                           -0.017850203824417415,-0.03507851122782909]

## Fitting GLMs with sparse matrices
srand(1)
X = sprand(1000, 10, 0.01)
β = randn(10)
y = Bool[rand() < logistic(x) for x in X * β]

gmsparse = fit(GeneralizedLinearModel, X, y, Binomial())
gmdense = fit(GeneralizedLinearModel, full(X), y, Binomial())

@test_approx_eq deviance(gmsparse) deviance(gmdense)
@test_approx_eq coef(gmsparse) coef(gmdense)
@test_approx_eq vcov(gmsparse) vcov(gmdense)

## Prediction for GLMs
srand(1)
X = rand(10, 2)
Y = Vector{Float64}(logistic.(X * [3; -3])) # Julia 0.4 loses type information so Vector{Float64} can be dropped when we don't support 0.4

gm11 = fit(GeneralizedLinearModel, X, Y, Binomial())
@test_approx_eq predict(gm11) Y

newX = rand(5, 2)
newY = Vector{Float64}(logistic.(newX * coef(gm11))) # Julia 0.4 loses type information so Vector{Float64} can be dropped when we don't support 0.4
@test_approx_eq predict(gm11, newX) newY

off = rand(10)
newoff = rand(5)

@test_throws ArgumentError predict(gm11, newX, offset=newoff)

gm12 = fit(GeneralizedLinearModel, X, Y, Binomial(), offset=off)
@test_throws ArgumentError predict(gm12, newX)
@test_approx_eq predict(gm12, newX, offset=newoff) logistic.(newX * coef(gm12) .+ newoff)

## Prediction from DataFrames
d = convert(DataFrame, X)
d[:y] = Y

gm13 = fit(GeneralizedLinearModel, y ~ 0 + x1 + x2, d, Binomial())
@test predict(gm13) == predict(gm13, d[[:x1, :x2]])
@test predict(gm13) == predict(gm13, d)

newd = convert(DataFrame, newX)
predict(gm13, newd)

# Issue 118
@inferred nobs(lm(randn(10, 2), randn(10)))

# Issue 84
let
    X=[1 1; 2 4; 3 9]
    Xf=[1 1; 2 4; 3 9.]
    y = [2, 6, 12]
    yf = [2, 6, 12.]
    @test_approx_eq(lm(X, y).pp.beta0, ones(2))
    @test_approx_eq(lm(Xf, y).pp.beta0, ones(2))
    @test_approx_eq(lm(X, yf).pp.beta0, ones(2))
end
