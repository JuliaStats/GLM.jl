using Base.Test, StatsFuns, DataFrames, GLM

function test_show(x)
    io = IOBuffer()
    show(io, x)
end

const glm_datadir = joinpath(dirname(@__FILE__), "..", "data")

## Formaldehyde data from the R Datasets package
form = DataFrame(Any[[0.1,0.3,0.5,0.6,0.7,0.9],[0.086,0.269,0.446,0.538,0.626,0.782]],
    [:Carb, :OptDen])

@testset "lm" begin
    lm1 = fit(LinearModel, OptDen ~ Carb, form)
    test_show(lm1)
    @test isapprox(coef(lm1), collect(linreg(form[:Carb], form[:OptDen])))
    Σ = [6.136653061224592e-05 -9.464489795918525e-05
        -9.464489795918525e-05 1.831836734693908e-04]
    @test isapprox(vcov(lm1), Σ)
    @test isapprox(cor(lm1.model), diagm(diag(Σ))^(-1/2)*Σ*diagm(diag(Σ))^(-1/2))
    @test dof(lm1) == 3
    @test isapprox(deviance(lm1), 0.0002992000000000012)
    @test isapprox(loglikelihood(lm1), 21.204842144047973)
    @test isapprox(nulldeviance(lm1), 0.3138488333333334)
    @test isapprox(nullloglikelihood(lm1), 0.33817870295676444)
    @test r²(lm1) == r2(lm1)
    @test isapprox(r²(lm1), 0.9990466748057584)
    @test adjr²(lm1) == adjr2(lm1)
    @test isapprox(adjr²(lm1), 0.998808343507198)
    @test isapprox(aic(lm1), -36.409684288095946)
    @test isapprox(aicc(lm1), -24.409684288095946)
    @test isapprox(bic(lm1), -37.03440588041178)
end

dobson = DataFrame(Counts = [18.,17,15,20,10,20,25,13,12],
    Outcome = pool(repeat(["A", "B", "C"], outer = 3)),
    Treatment = pool(repeat(["a","b", "c"], inner = 3)))

@testset "Poisson GLM" begin
    gm1 = fit(GeneralizedLinearModel, Counts ~ Outcome + Treatment, dobson, Poisson())
    test_show(gm1)
    @test dof(gm1) == 5
    @test isapprox(deviance(gm1), 5.12914107700115, rtol = 1e-7)
    @test isapprox(loglikelihood(gm1), -23.380659200978837, rtol = 1e-7)
    @test isapprox(aic(gm1), 56.76131840195767)
    @test isapprox(aicc(gm1), 76.76131840195768)
    @test isapprox(bic(gm1), 57.74744128863877)
    @test isapprox(coef(gm1)[1:3],
        [3.044522437723423,-0.45425527227759555,-0.29298712468147375])
end

## Example from http://www.ats.ucla.edu/stat/r/dae/logit.htm
admit = readtable(joinpath(glm_datadir, "admit.csv.gz"))
admit[:rank] = pool(admit[:rank])

@testset "Binomial, Bernoulli, LogitLink" begin
    for distr in (Binomial, Bernoulli)
        gm2 = fit(GeneralizedLinearModel, admit ~ gre + gpa + rank, admit, distr())
        test_show(gm2)
        @test dof(gm2) == 6
        @test isapprox(deviance(gm2), 458.5174924758994)
        @test isapprox(loglikelihood(gm2), -229.25874623794968)
        @test isapprox(aic(gm2), 470.51749247589936)
        @test isapprox(aicc(gm2), 470.7312329339146)
        @test isapprox(bic(gm2), 494.4662797585473)
        @test isapprox(coef(gm2),
            [-3.9899786606380734, 0.0022644256521549043, 0.8040374535155766,
            -0.6754428594116577, -1.3402038117481079,-1.5514636444657492])
    end
end

@testset "Bernoulli ProbitLink" begin
    gm3 = fit(GeneralizedLinearModel, admit ~ gre + gpa + rank, admit,
        Binomial(), ProbitLink())
    test_show(gm3)
    @test dof(gm3) == 6
    @test isapprox(deviance(gm3), 458.4131713833386)
    @test isapprox(loglikelihood(gm3), -229.20658569166932)
    @test isapprox(aic(gm3), 470.41317138333864)
    @test isapprox(aicc(gm3), 470.6269118413539)
    @test isapprox(bic(gm3), 494.36195866598655)
    @test isapprox(coef(gm3),
        [-2.3867922998680786, 0.0013755394922972369, 0.47772908362647015,
        -0.4154125854823675, -0.8121458010130356, -0.9359047862425298])
end

@testset "Bernoulli CauchitLink" begin
    gm4 = fit(GeneralizedLinearModel, admit ~ gre + gpa + rank, admit,
        Binomial(), CauchitLink())
    test_show(gm4)
    @test dof(gm4) == 6
    @test isapprox(deviance(gm4), 459.3401112751141)
    @test isapprox(loglikelihood(gm4), -229.6700556375571)
    @test isapprox(aic(gm4), 471.3401112751142)
    @test isapprox(aicc(gm4), 471.5538517331295)
    @test isapprox(bic(gm4), 495.28889855776214)
end

@testset "Bernoulli CloglogLink" begin
    gm5 = fit(GeneralizedLinearModel, admit ~ gre + gpa + rank, admit,
        Binomial(), CloglogLink())
    test_show(gm5)
    @test dof(gm5) == 6
    @test isapprox(deviance(gm5), 458.89439629612616)
    @test isapprox(loglikelihood(gm5), -229.44719814806314)
    @test isapprox(aic(gm5), 470.8943962961263)
    @test isapprox(aicc(gm5), 471.1081367541415)
    @test isapprox(bic(gm5), 494.8431835787742)
end

## Example with offsets from Venables & Ripley (2002, p.189)
anorexia = readtable(joinpath(glm_datadir, "anorexia.csv.gz"))
anorexia[:Treat] = pool(anorexia[:Treat])

@testset "Offset" begin
    gm6 = fit(GeneralizedLinearModel, Postwt ~ Prewt + Treat, anorexia,
        Normal(), IdentityLink(), offset=Array(anorexia[:Prewt]))
    test_show(gm6)
    @test dof(gm6) == 5
    @test isapprox(deviance(gm6), 3311.262619919613)
    @test isapprox(loglikelihood(gm6), -239.9866487711122)
    @test isapprox(aic(gm6), 489.9732975422244)
    @test isapprox(aicc(gm6), 490.8823884513153)
    @test isapprox(bic(gm6), 501.35662813730465)
    @test isapprox(coef(gm6),
        [49.7711090149846,-0.5655388496391,-4.0970655280729,4.5630626529188])
    @test isapprox(GLM.dispersion(gm6.model, true), 48.6950385282296)
    @test isapprox(stderr(gm6),
        [13.3909581420259, 0.1611823618518, 1.8934926069669, 2.1333359226431])
end

@testset "Normal LogLink offset" begin
    gm7 = fit(GeneralizedLinearModel, Postwt ~ Prewt + Treat, anorexia,
        Normal(), LogLink(), offset=Array(anorexia[:Prewt]), convTol=1e-8)
    test_show(gm7)
    @test isapprox(deviance(gm7), 3265.207242977156)
    @test isapprox(coef(gm7),
        [3.992326787835955, -0.994452693131178, -0.050698258703974, 0.051494029957641])
    @test isapprox(GLM.dispersion(gm7.model, true), 48.017753573192266)
    @test isapprox(stderr(gm7), [0.15716774, 0.0018862835, 0.02258404, 0.023882795], atol=1e-6)
end

## Gamma example from McCullagh & Nelder (1989, pp. 300-2)
clotting = DataFrame(u = log.([5,10,15,20,30,40,60,80,100]),
                     lot1 = [118,58,42,35,27,25,21,19,18])

@testset "Gamma InverseLink" begin
    gm8 = fit(GeneralizedLinearModel, lot1 ~ 1 + u, clotting, Gamma())
    test_show(gm8)
    @test dof(gm8) == 3
    @test isapprox(deviance(gm8), 0.016729715178484157)
    @test isapprox(loglikelihood(gm8), -15.994961974777247)
    @test isapprox(aic(gm8), 37.989923949554495)
    @test isapprox(aicc(gm8), 42.78992394955449)
    @test isapprox(bic(gm8), 38.58159768156315)
    @test isapprox(coef(gm8), [-0.01655438172784895,0.01534311491072141])
    @test isapprox(GLM.dispersion(gm8.model, true), 0.002446059333495581, atol=1e-6)
    @test isapprox(stderr(gm8), [0.00092754223, 0.000414957683], atol=1e-6)
end

@testset "Gamma LogLink" begin
    gm9 = fit(GeneralizedLinearModel, lot1 ~ 1 + u, clotting, Gamma(), LogLink(),
        convTol=1e-8)
    test_show(gm9)
    @test dof(gm9) == 3
    @test isapprox(deviance(gm9), 0.16260829451739)
    @test isapprox(loglikelihood(gm9), -26.24082810384911)
    @test isapprox(aic(gm9), 58.48165620769822)
    @test isapprox(aicc(gm9), 63.28165620769822)
    @test isapprox(bic(gm9), 59.07332993970688)
    @test isapprox(coef(gm9), [5.50322528458221, -0.60191617825971])
    @test isapprox(GLM.dispersion(gm9.model, true), 0.02435442293561081)
    @test isapprox(stderr(gm9), [0.19030107482720, 0.05530784660144])
end

@testset "Gamma IdentityLink" begin
    gm10 = fit(GeneralizedLinearModel, lot1 ~ 1 + u, clotting, Gamma(), IdentityLink(),
        convTol=1e-8)
    test_show(gm10)
    @test dof(gm10) == 3
    @test isapprox(deviance(gm10), 0.60845414895344)
    @test isapprox(loglikelihood(gm10), -32.216072437284176)
    @test isapprox(aic(gm10), 70.43214487456835)
    @test isapprox(aicc(gm10), 75.23214487456835)
    @test isapprox(bic(gm10), 71.02381860657701)
    @test isapprox(coef(gm10), [99.250446880986, -18.374324929002])
    @test isapprox(GLM.dispersion(gm10.model, true), 0.10417373, atol=1e-6)
    @test isapprox(stderr(gm10), [17.864084, 4.297895], atol=1e-4)
end

# Logistic regression using aggregated data and weights
admit_agr = DataFrame(count = [28., 97, 93, 55, 33, 54, 28, 12],
                      admit = repeat([false, true], inner=[4]),
                      rank = pool(repeat([1, 2, 3, 4], outer=[2])))

@testset "Aggregated Binomial LogitLink" begin
    for distr in (Binomial, Bernoulli)
        gm14 = fit(GeneralizedLinearModel, admit ~ rank, admit_agr, distr(), wts=Array(admit_agr[:count]))
        @test dof(gm14) == 4
        @test nobs(gm14) == 400
        @test isapprox(deviance(gm14), 474.9667184280627)
        @test isapprox(loglikelihood(gm14), -237.48335921403134)
        @test isapprox(aic(gm14), 482.96671842822883)
        @test isapprox(aicc(gm14), 483.0679842510136)
        @test isapprox(bic(gm14), 498.9325766164946)
        @test isapprox(coef(gm14),
            [0.164303051291, -0.7500299832, -1.36469792994, -1.68672866457], atol=1e-5)
    end
end

# Logistic regression using aggregated data with proportions of successes and weights
admit_agr2 = DataFrame(Any[[61., 151, 121, 67], [33., 54, 28, 12], pool([1, 2, 3, 4])],
    [:count, :admit, :rank])
admit_agr2[:p] = admit_agr2[:admit] ./ admit_agr2[:count]

## The model matrix here is singular so tests like the deviance are just round off error
@testset "Binomial LogitLink aggregated" begin
    gm15 = fit(GeneralizedLinearModel, p ~ rank, admit_agr2, Binomial(), wts=admit_agr2[:count])
    test_show(gm15)
    @test dof(gm15) == 4
    @test nobs(gm15) == 400
# The model matrix is singular so the deviance is essentially round-off error
#    @test isapprox(deviance(gm15), -2.4424906541753456e-15, rtol = 1e-7)
    @test isapprox(loglikelihood(gm15), -9.50254433604239)
    @test isapprox(aic(gm15), 27.00508867208478)
    @test isapprox(aicc(gm15), 27.106354494869592)
    @test isapprox(bic(gm15), 42.970946860516705)
    @test isapprox(coef(gm15),
        [0.1643030512912767, -0.7500299832303851, -1.3646980342693287, -1.6867295867357475])
end

# Weighted Gamma example (weights are totally made up)
@testset "Gamma InverseLink Weights" begin
    gm16 = fit(GeneralizedLinearModel, lot1 ~ 1 + u, clotting, Gamma(),
        wts=[1.5,2.0,1.1,4.5,2.4,3.5,5.6,5.4,6.7])
    test_show(gm16)
    @test dof(gm16) == 3
    @test nobs(gm16) == 32.7
    @test isapprox(deviance(gm16), 0.03933389380881689)
    @test isapprox(loglikelihood(gm16), -43.35907878769152)
    @test isapprox(aic(gm16), 92.71815757538305)
    @test isapprox(aicc(gm16), 93.55439450918095)
    @test isapprox(bic(gm16), 97.18028280909267)
    @test isapprox(coef(gm16), [-0.017217012615523237, 0.015649040411276433])
end

# Weighted Poisson example (weights are totally made up)
@testset "Poisson LogLink Weights" begin
    gm17 = fit(GeneralizedLinearModel, Counts ~ Outcome + Treatment, dobson, Poisson(),
        wts = [1.5,2.0,1.1,4.5,2.4,3.5,5.6,5.4,6.7])
    test_show(gm17)
    @test dof(gm17) == 5
    @test isapprox(deviance(gm17), 17.699857821414266)
    @test isapprox(loglikelihood(gm17), -84.57429468506352)
    @test isapprox(aic(gm17), 179.14858937012704)
    @test isapprox(aicc(gm17), 181.39578038136298)
    @test isapprox(bic(gm17), 186.5854647596431)
    @test isapprox(coef(gm17), [3.1218557035404793, -0.5270435906931427,-0.40300384148562746,
                           -0.017850203824417415,-0.03507851122782909])
end

@testset "Sparse GLM" begin
    srand(1)
    X = sprand(1000, 10, 0.01)
    β = randn(10)
    y = Bool[rand() < logistic(x) for x in X * β]
    gmsparse = fit(GeneralizedLinearModel, X, y, Binomial())
    gmdense = fit(GeneralizedLinearModel, full(X), y, Binomial())

    @test isapprox(deviance(gmsparse), deviance(gmdense))
    @test isapprox(coef(gmsparse), coef(gmdense))
    @test isapprox(vcov(gmsparse), vcov(gmdense))
end


@testset "Predict" begin
    srand(1)
    X = rand(10, 2)
    Y = logistic.(X * [3; -3])

    gm11 = fit(GeneralizedLinearModel, X, Y, Binomial())
    @test isapprox(predict(gm11), Y)

    newX = rand(5, 2)
    newY = logistic.(newX * coef(gm11))
    @test isapprox(predict(gm11, newX), newY)

    off = rand(10)
    newoff = rand(5)

    @test_throws ArgumentError predict(gm11, newX, offset=newoff)

    gm12 = fit(GeneralizedLinearModel, X, Y, Binomial(), offset=off)
    @test_throws ArgumentError predict(gm12, newX)
    @test isapprox(predict(gm12, newX, offset=newoff),
        logistic.(newX * coef(gm12) .+ newoff))

        # Prediction from DataFrames
    d = convert(DataFrame, X)
    d[:y] = Y

    gm13 = fit(GeneralizedLinearModel, y ~ 0 + x1 + x2, d, Binomial())
    @test predict(gm13) == predict(gm13, d[[:x1, :x2]])
    @test predict(gm13) == predict(gm13, d)

    newd = convert(DataFrame, newX)
    predict(gm13, newd)
end

@testset "Issue 118" begin
    @inferred nobs(lm(randn(10, 2), randn(10)))
end

@testset "Issue 84" begin
    X = [1 1; 2 4; 3 9]
    Xf = [1 1; 2 4; 3 9.]
    y = [2, 6, 12]
    yf = [2, 6, 12.]
    @test isapprox(lm(X, y).pp.beta0, ones(2))
    @test isapprox(lm(Xf, y).pp.beta0, ones(2))
    @test isapprox(lm(X, yf).pp.beta0, ones(2))
end

@testset "F test for model comparison" begin
    d = DataFrame(Treatment=[1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2.],
                  Result=[1.1, 1.2, 1, 2.2, 1.9, 2, .9, 1, 1, 2.2, 2, 2],
                  Other=[1, 1, 2, 1, 2, 1, 3, 1, 1, 2, 2, 1])
    mod = lm(@formula(Result~Treatment), d).model
    othermod = lm(@formula(Result~Other), d).model
    nullmod = lm(@formula(Result~1), d).model
    @test GLM.issubmodel(nullmod, mod)
    @test !GLM.issubmodel(othermod, mod)
    
    @test_throws ArgumentError ftest(mod, othermod)
    
    ft = ftest(mod, nullmod)
    @test isapprox(ft.cols[end][2],  2.481215056713184e-8)
    # Test output
    @test sprint(show, ftest(mod, nullmod)) == 
        """
                 Res. DOF DOF ΔDOF      SSR     ΔSSR           R²      ΔR²      F*      p(>F)
        Model 1      10.0 3.0  NaN 0.128333      NaN     0.960258      NaN     NaN        NaN
        Model 2      11.0 2.0  1.0  3.22917 -3.10083 -2.22045e-16 0.960258 241.623 2.48122e-8
        """
    
    bigmod = lm(@formula(Result~Treatment+Other), d).model
    ft2 = ftest(bigmod, mod, nullmod)
    @test isapprox(ft2.cols[end][3],  2.481215056713184e-8)
    @test isapprox(ft2.cols[end][2], 0.17903437900958952)
    @test sprint(show, ftest(bigmod, mod, nullmod)) == 
        """
                 Res. DOF DOF ΔDOF      SSR     ΔSSR           R²       ΔR²      F*      p(>F)
        Model 1       9.0 4.0  NaN 0.103833      NaN     0.967845       NaN     NaN        NaN
        Model 2      10.0 3.0  1.0 0.128333  -0.0245     0.960258 0.0075871  2.1236   0.179034
        Model 3      11.0 2.0  1.0  3.22917 -3.10083 -2.22045e-16  0.960258 241.623 2.48122e-8
        """
end
