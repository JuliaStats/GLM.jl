using CategoricalArrays, CSV, DataFrames, LinearAlgebra, SparseArrays, StableRNGs,
      Statistics, StatsBase, Test, RDatasets
using GLM
using StatsFuns: logistic
using Distributions: TDist

test_show(x) = show(IOBuffer(), x)

const glm_datadir = joinpath(dirname(@__FILE__), "..", "data")

## Formaldehyde data from the R Datasets package
form = DataFrame([[0.1,0.3,0.5,0.6,0.7,0.9],[0.086,0.269,0.446,0.538,0.626,0.782]],
    [:Carb, :OptDen])

function simplemm(x::AbstractVecOrMat)
    n = size(x, 2)
    mat = fill(one(float(eltype(x))), length(x), n + 1)
    copyto!(view(mat, :, 2:(n + 1)), x)
    mat
end

linreg(x::AbstractVecOrMat, y::AbstractVector) = qr!(simplemm(x)) \ y

@testset "lm" begin
    lm1 = fit(LinearModel, @formula(OptDen ~ Carb), form)
    test_show(lm1)
    @test isapprox(coef(lm1), linreg(form.Carb, form.OptDen))
    Σ = [6.136653061224592e-05 -9.464489795918525e-05
        -9.464489795918525e-05 1.831836734693908e-04]
    @test isapprox(vcov(lm1), Σ)
    @test isapprox(cor(lm1.model), Diagonal(diag(Σ))^(-1/2)*Σ*Diagonal(diag(Σ))^(-1/2))
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
    lm2 = fit(LinearModel, hcat(ones(6), 10form.Carb), form.OptDen, true)
    @test isa(lm2.pp.chol, CholeskyPivoted)
    @test lm2.pp.chol.piv == [2, 1]
    @test isapprox(coef(lm1), coef(lm2) .* [1., 10.])
    lm3 = lm(@formula(y~x), (y=1:25, x=repeat(1:5, 5)), contrasts=Dict(:x=>DummyCoding()))
    lm4 = lm(@formula(y~x), (y=1:25, x=categorical(repeat(1:5, 5))))
    @test coef(lm3) == coef(lm4) ≈ [11, 1, 2, 3, 4]
end

@testset "Linear Model Cook's Distance" begin
    st_df = DataFrame( 
        Y=[6.4, 7.4, 10.4, 15.1, 12.3 , 11.4],
        XA=[1.5, 6.5, 11.5, 19.9, 17.0, 15.5],
        XB=[1.8, 7.8, 11.8, 20.5, 17.3, 15.8], 
        XC=[3., 13., 23., 39.8, 34., 31.],
        # values from SAS proc reg
        CooksD_base=[1.4068501943, 0.176809102, 0.0026655177, 1.0704009915, 0.0875726457, 0.1331183932], 
        CooksD_noint=[0.0076891801, 0.0302993877, 0.0410262965, 0.0294348488, 0.0691589296, 0.0273045538], 
        CooksD_multi=[1.7122291956, 18.983407026, 0.000118078, 0.8470797843, 0.0715921999, 0.1105843157],
        )

    # linear regression
    t_lm_base = lm(@formula(Y ~ XA), st_df)
    @test isapprox(st_df.CooksD_base, cooksdistance(t_lm_base))

    # linear regression, no intercept 
    t_lm_noint = lm(@formula(Y ~ XA +0), st_df)
    @test isapprox(st_df.CooksD_noint, cooksdistance(t_lm_noint))

    # linear regression, two collinear variables (Variance inflation factor ≊ 250)
    t_lm_multi = lm(@formula(Y ~ XA + XB), st_df)
    @test isapprox(st_df.CooksD_multi, cooksdistance(t_lm_multi))

    # linear regression, two full collinear variables (XC = 2 XA) hence should get the same results as the original
    # after pivoting
    t_lm_colli = lm(@formula(Y ~ XA + XC), st_df, dropcollinear=true)
    # Currently fails as the collinear variable is not dropped from `modelmatrix(obj)`
    @test_throws ArgumentError isapprox(st_df.CooksD_base, cooksdistance(t_lm_colli))

end

@testset "linear model with weights" begin 
    df = dataset("quantreg", "engel")
    N = nrow(df)
    df.weights = repeat(1:5, Int(N/5))
    f = @formula(FoodExp ~ Income)
    lm_model = lm(f, df, wts = df.weights)
    glm_model = glm(f, df, Normal(), wts = df.weights)
    @test isapprox(coef(lm_model), [154.35104595140706, 0.4836896390157505])
    @test isapprox(coef(glm_model), [154.35104595140706, 0.4836896390157505])
    @test isapprox(stderror(lm_model), [9.382302620120193, 0.00816741377772968])
    @test isapprox(r2(lm_model), 0.8330258148644486)
    @test isapprox(adjr2(lm_model), 0.832788298242634)
    @test isapprox(vcov(lm_model), [88.02760245551447 -0.06772589439264813; 
                                    -0.06772589439264813 6.670664781664879e-5])
    @test isapprox(first(predict(lm_model)), 357.57694841780994)
    @test isapprox(loglikelihood(lm_model), -4353.946729075838)
    @test isapprox(loglikelihood(glm_model), -4353.946729075838)
    @test isapprox(nullloglikelihood(lm_model), -4984.892139711452)
    @test isapprox(mean(residuals(lm_model)), -5.412966629787718) 
end

@testset "rankdeficient" begin
    rng = StableRNG(1234321)
    # an example of rank deficiency caused by a missing cell in a table
    dfrm = DataFrame([categorical(repeat(string.('A':'D'), inner = 6)),
                     categorical(repeat(string.('a':'c'), inner = 2, outer = 4))],
                     [:G, :H])
    f = @formula(0 ~ 1 + G*H)
    X = ModelMatrix(ModelFrame(f, dfrm)).m
    y = X * (1:size(X, 2)) + 0.1 * randn(rng, size(X, 1))
    inds = deleteat!(collect(1:length(y)), 7:8)
    m1 = fit(LinearModel, X, y)
    @test isapprox(deviance(m1), 0.12160301538297297)
    Xmissingcell = X[inds, :]
    ymissingcell = y[inds]
    @test_throws PosDefException m2 = fit(LinearModel, Xmissingcell, ymissingcell; dropcollinear=false)
    m2p = fit(LinearModel, Xmissingcell, ymissingcell)
    @test isa(m2p.pp.chol, CholeskyPivoted)
    @test rank(m2p.pp.chol) == 11
    @test isapprox(deviance(m2p), 0.1215758392280204)
    @test isapprox(coef(m2p), [0.9772643585228885, 8.903341608496437, 3.027347397503281,
        3.9661379199401257, 5.079410103608552, 6.1944618141188625, 0.0, 7.930328728005131,
        8.879994918604757, 2.986388408421915, 10.84972230524356, 11.844809275711485])

    m2p_dep_pos = fit(LinearModel, Xmissingcell, ymissingcell, true)
    @test_logs (:warn, "Positional argument `allowrankdeficient` is deprecated, use keyword " *
                "argument `dropcollinear` instead. Proceeding with positional argument value: true") fit(LinearModel, Xmissingcell, ymissingcell, true)
    @test isa(m2p_dep_pos.pp.chol, CholeskyPivoted)
    @test rank(m2p_dep_pos.pp.chol) == rank(m2p.pp.chol)
    @test isapprox(deviance(m2p_dep_pos), deviance(m2p))
    @test isapprox(coef(m2p_dep_pos), coef(m2p))

    m2p_dep_pos_kw = fit(LinearModel, Xmissingcell, ymissingcell, true; dropcollinear = false)
    @test isa(m2p_dep_pos_kw.pp.chol, CholeskyPivoted)
    @test rank(m2p_dep_pos_kw.pp.chol) == rank(m2p.pp.chol)
    @test isapprox(deviance(m2p_dep_pos_kw), deviance(m2p))
    @test isapprox(coef(m2p_dep_pos_kw), coef(m2p))
end

dobson = DataFrame(Counts = [18.,17,15,20,10,20,25,13,12],
    Outcome = categorical(repeat(string.('A':'C'), outer = 3)),
    Treatment = categorical(repeat(string.('a':'c'), inner = 3)))

@testset "Poisson GLM" begin
    gm1 = fit(GeneralizedLinearModel, @formula(Counts ~ 1 + Outcome + Treatment),
              dobson, Poisson())
    @test GLM.cancancel(gm1.model.rr)
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
admit = CSV.read(joinpath(glm_datadir, "admit.csv"), DataFrame)
admit.rank = categorical(admit.rank)

@testset "$distr with LogitLink" for distr in (Binomial, Bernoulli)
    gm2 = fit(GeneralizedLinearModel, @formula(admit ~ 1 + gre + gpa + rank), admit, distr())
    @test GLM.cancancel(gm2.model.rr)
    test_show(gm2)
    @test dof(gm2) == 6
    @test deviance(gm2) ≈ 458.5174924758994
    @test loglikelihood(gm2) ≈ -229.25874623794968
    @test isapprox(aic(gm2), 470.51749247589936)
    @test isapprox(aicc(gm2), 470.7312329339146)
    @test isapprox(bic(gm2), 494.4662797585473)
    @test isapprox(coef(gm2),
        [-3.9899786606380756, 0.0022644256521549004, 0.804037453515578,
         -0.6754428594116578, -1.340203811748108, -1.5514636444657495])
end

@testset "Bernoulli ProbitLink" begin
    gm3 = fit(GeneralizedLinearModel, @formula(admit ~ 1 + gre + gpa + rank), admit,
              Binomial(), ProbitLink())
    test_show(gm3)
    @test !GLM.cancancel(gm3.model.rr)
    @test dof(gm3) == 6
    @test isapprox(deviance(gm3), 458.4131713833386)
    @test isapprox(loglikelihood(gm3), -229.20658569166932)
    @test isapprox(aic(gm3), 470.41317138333864)
    @test isapprox(aicc(gm3), 470.6269118413539)
    @test isapprox(bic(gm3), 494.36195866598655)
    @test isapprox(coef(gm3),
        [-2.3867922998680777, 0.0013755394922972401, 0.47772908362646926,
        -0.4154125854823675, -0.8121458010130354, -0.9359047862425297])
end

@testset "Bernoulli CauchitLink" begin
    gm4 = fit(GeneralizedLinearModel, @formula(admit ~ gre + gpa + rank), admit,
              Binomial(), CauchitLink())
    @test !GLM.cancancel(gm4.model.rr)
    test_show(gm4)
    @test dof(gm4) == 6
    @test isapprox(deviance(gm4), 459.3401112751141)
    @test isapprox(loglikelihood(gm4), -229.6700556375571)
    @test isapprox(aic(gm4), 471.3401112751142)
    @test isapprox(aicc(gm4), 471.5538517331295)
    @test isapprox(bic(gm4), 495.28889855776214)
end

@testset "Bernoulli CloglogLink" begin
    gm5 = fit(GeneralizedLinearModel, @formula(admit ~ gre + gpa + rank), admit,
              Binomial(), CloglogLink())
    @test !GLM.cancancel(gm5.model.rr)
    test_show(gm5)
    @test dof(gm5) == 6
    @test isapprox(deviance(gm5), 458.89439629612616)
    @test isapprox(loglikelihood(gm5), -229.44719814806314)
    @test isapprox(aic(gm5), 470.8943962961263)
    @test isapprox(aicc(gm5), 471.1081367541415)
    @test isapprox(bic(gm5), 494.8431835787742)

    # When data are almost separated, the calculations are prone to underflow which can cause
    # NaN in wrkwt and/or wrkres. The example here used to fail but works with the "clamping"
    # introduced in #187
    @testset "separated data" begin
        n   = 100
        rng = StableRNG(127)

        X = [ones(n) randn(rng, n)]
        y = logistic.(X*ones(2) + 1/10*randn(rng, n)) .> 1/2
        @test coeftable(glm(X, y, Binomial(), CloglogLink())).cols[4][2] < 0.05
    end
end

## Example with offsets from Venables & Ripley (2002, p.189)
anorexia = CSV.read(joinpath(glm_datadir, "anorexia.csv"), DataFrame)

@testset "Offset" begin
    gm6 = fit(GeneralizedLinearModel, @formula(Postwt ~ 1 + Prewt + Treat), anorexia,
              Normal(), IdentityLink(), offset=Array{Float64}(anorexia.Prewt))
    @test GLM.cancancel(gm6.model.rr)
    test_show(gm6)
    @test dof(gm6) == 5
    @test isapprox(deviance(gm6), 3311.262619919613)
    @test isapprox(loglikelihood(gm6), -239.9866487711122)
    @test isapprox(aic(gm6), 489.9732975422244)
    @test isapprox(aicc(gm6), 490.8823884513153)
    @test isapprox(bic(gm6), 501.35662813730465)
    @test isapprox(coef(gm6),
        [49.7711090, -0.5655388, -4.0970655, 4.5630627])
    @test isapprox(GLM.dispersion(gm6.model, true), 48.6950385282296)
    @test isapprox(stderror(gm6),
        [13.3909581, 0.1611824, 1.8934926, 2.1333359])
end

@testset "Normal LogLink offset" begin
    gm7 = fit(GeneralizedLinearModel, @formula(Postwt ~ 1 + Prewt + Treat), anorexia,
              Normal(), LogLink(), offset=Array{Float64}(anorexia.Prewt), rtol=1e-8)
    @test !GLM.cancancel(gm7.model.rr)
    test_show(gm7)
    @test isapprox(deviance(gm7), 3265.207242977156)
    @test isapprox(coef(gm7),
        [3.99232679, -0.99445269, -0.05069826, 0.05149403])
    @test isapprox(GLM.dispersion(gm7.model, true), 48.017753573192266)
    @test isapprox(stderror(gm7),
        [0.157167944, 0.001886286, 0.022584069, 0.023882826],
        atol=1e-6)
end

## Gamma example from McCullagh & Nelder (1989, pp. 300-2)
clotting = DataFrame(u = log.([5,10,15,20,30,40,60,80,100]),
                     lot1 = [118,58,42,35,27,25,21,19,18])

@testset "Gamma" begin
    gm8 = fit(GeneralizedLinearModel, @formula(lot1 ~ 1 + u), clotting, Gamma())
    @test !GLM.cancancel(gm8.model.rr)
    @test isa(GLM.Link(gm8.model), InverseLink)
    test_show(gm8)
    @test dof(gm8) == 3
    @test isapprox(deviance(gm8), 0.016729715178484157)
    @test isapprox(loglikelihood(gm8), -15.994961974777247)
    @test isapprox(aic(gm8), 37.989923949554495)
    @test isapprox(aicc(gm8), 42.78992394955449)
    @test isapprox(bic(gm8), 38.58159768156315)
    @test isapprox(coef(gm8), [-0.01655438172784895,0.01534311491072141])
    @test isapprox(GLM.dispersion(gm8.model, true), 0.002446059333495581, atol=1e-6)
    @test isapprox(stderror(gm8), [0.00092754223, 0.000414957683], atol=1e-6)
end

@testset "InverseGaussian" begin
    gm8a = fit(GeneralizedLinearModel, @formula(lot1 ~ 1 + u), clotting, InverseGaussian())
    @test !GLM.cancancel(gm8a.model.rr)
    @test isa(GLM.Link(gm8a.model), InverseSquareLink)
    test_show(gm8a)
    @test dof(gm8a) == 3
    @test isapprox(deviance(gm8a), 0.006931128347234519)
    @test isapprox(loglikelihood(gm8a), -27.787426008849867)
    @test isapprox(aic(gm8a), 61.57485201769973)
    @test isapprox(aicc(gm8a), 66.37485201769974)
    @test isapprox(bic(gm8a), 62.16652574970839)
    @test isapprox(coef(gm8a), [-0.0011079770504295668,0.0007219138982289362])
    @test isapprox(GLM.dispersion(gm8a.model, true), 0.0011008719709455776, atol=1e-6)
    @test isapprox(stderror(gm8a), [0.0001675339726910311,9.468485015919463e-5], atol=1e-6)
end

@testset "Gamma LogLink" begin
    gm9 = fit(GeneralizedLinearModel, @formula(lot1 ~ 1 + u), clotting, Gamma(), LogLink(),
              rtol=1e-8, atol=0.0)
    @test !GLM.cancancel(gm9.model.rr)
    test_show(gm9)
    @test dof(gm9) == 3
    @test deviance(gm9) ≈ 0.16260829451739
    @test loglikelihood(gm9) ≈ -26.24082810384911
    @test aic(gm9) ≈ 58.48165620769822
    @test aicc(gm9) ≈ 63.28165620769822
    @test bic(gm9) ≈ 59.07332993970688
    @test coef(gm9) ≈ [5.50322528458221, -0.60191617825971]
    @test GLM.dispersion(gm9.model, true) ≈ 0.02435442293561081
    @test stderror(gm9) ≈ [0.19030107482720, 0.05530784660144]
end

@testset "Gamma IdentityLink" begin
    gm10 = fit(GeneralizedLinearModel, @formula(lot1 ~ 1 + u), clotting, Gamma(), IdentityLink(),
               rtol=1e-8, atol=0.0)
    @test !GLM.cancancel(gm10.model.rr)
    test_show(gm10)
    @test dof(gm10) == 3
    @test isapprox(deviance(gm10), 0.60845414895344)
    @test isapprox(loglikelihood(gm10), -32.216072437284176)
    @test isapprox(aic(gm10), 70.43214487456835)
    @test isapprox(aicc(gm10), 75.23214487456835)
    @test isapprox(bic(gm10), 71.02381860657701)
    @test isapprox(coef(gm10), [99.250446880986, -18.374324929002])
    @test isapprox(GLM.dispersion(gm10.model, true), 0.10417373, atol=1e-6)
    @test isapprox(stderror(gm10), [17.864084, 4.297895], atol=1e-4)
end

# Logistic regression using aggregated data and weights
admit_agr = DataFrame(count = [28., 97, 93, 55, 33, 54, 28, 12],
                      admit = repeat([false, true], inner=[4]),
                      rank = categorical(repeat(1:4, outer=2)))

@testset "Aggregated Binomial LogitLink" begin
    for distr in (Binomial, Bernoulli)
        gm14 = fit(GeneralizedLinearModel, @formula(admit ~ 1 + rank), admit_agr, distr(),
                   wts=Array(admit_agr.count))
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
admit_agr2 = DataFrame(Any[[61., 151, 121, 67], [33., 54, 28, 12], categorical(1:4)],
    [:count, :admit, :rank])
admit_agr2.p = admit_agr2.admit ./ admit_agr2.count

## The model matrix here is singular so tests like the deviance are just round off error
@testset "Binomial LogitLink aggregated" begin
    gm15 = fit(GeneralizedLinearModel, @formula(p ~ rank), admit_agr2, Binomial(),
               wts=admit_agr2.count)
    test_show(gm15)
    @test dof(gm15) == 4
    @test nobs(gm15) == 400
    @test deviance(gm15) ≈ -2.4424906541753456e-15 atol = 1e-13
    @test loglikelihood(gm15) ≈ -9.50254433604239
    @test aic(gm15) ≈ 27.00508867208478
    @test aicc(gm15) ≈ 27.106354494869592
    @test bic(gm15) ≈ 42.970946860516705
    @test coef(gm15) ≈ [0.1643030512912767, -0.7500299832303851, -1.3646980342693287, -1.6867295867357475]
end

# Weighted Gamma example (weights are totally made up)
@testset "Gamma InverseLink Weights" begin
    gm16 = fit(GeneralizedLinearModel, @formula(lot1 ~ 1 + u), clotting, Gamma(),
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
    gm17 = fit(GeneralizedLinearModel, @formula(Counts ~ Outcome + Treatment), dobson, Poisson(),
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

# "quine" dataset discussed in Section 7.4 of "Modern Applied Statistics with S"
quine = dataset("MASS", "quine")
@testset "NegativeBinomial LogLink Fixed θ" begin
    gm18 = fit(GeneralizedLinearModel, @formula(Days ~ Eth+Sex+Age+Lrn), quine, NegativeBinomial(2.0), LogLink())
    @test !GLM.cancancel(gm18.model.rr)
    test_show(gm18)
    @test dof(gm18) == 8
    @test isapprox(deviance(gm18), 239.11105911824325, rtol = 1e-7)
    @test isapprox(loglikelihood(gm18), -553.2596040803376, rtol = 1e-7)
    @test isapprox(aic(gm18), 1122.5192081606751)
    @test isapprox(aicc(gm18), 1123.570303051186)
    @test isapprox(bic(gm18), 1146.3880611343418)
    @test isapprox(coef(gm18)[1:7],
        [2.886448718885344, -0.5675149923412003, 0.08707706381784373,
        -0.44507646428307207, 0.09279987988262384, 0.35948527963485755, 0.29676767190444386])
end

@testset "NegativeBinomial NegativeBinomialLink Fixed θ" begin
    # the default/canonical link is NegativeBinomialLink
    gm19 = fit(GeneralizedLinearModel, @formula(Days ~ Eth+Sex+Age+Lrn), quine, NegativeBinomial(2.0))
    @test GLM.cancancel(gm19.model.rr)
    test_show(gm19)
    @test dof(gm19) == 8
    @test isapprox(deviance(gm19), 239.68562048977307, rtol = 1e-7)
    @test isapprox(loglikelihood(gm19), -553.5468847661017, rtol = 1e-7)
    @test isapprox(aic(gm19), 1123.0937695322034)
    @test isapprox(aicc(gm19), 1124.1448644227144)
    @test isapprox(bic(gm19), 1146.96262250587)
    @test isapprox(coef(gm19)[1:7],
        [-0.12737182842213654, -0.055871700989224705, 0.01561618806384601,
        -0.041113722732799125, 0.024042387142113462, 0.04400234618798099, 0.035765875508382027,
])
end

@testset "NegativeBinomial LogLink, θ to be estimated" begin
    gm20 = negbin(@formula(Days ~ Eth+Sex+Age+Lrn), quine, LogLink())
    test_show(gm20)
    @test dof(gm20) == 8
    @test isapprox(deviance(gm20), 167.9518430624193, rtol = 1e-7)
    @test isapprox(loglikelihood(gm20), -546.57550938017, rtol = 1e-7)
    @test isapprox(aic(gm20), 1109.15101876034)
    @test isapprox(aicc(gm20), 1110.202113650851)
    @test isapprox(bic(gm20), 1133.0198717340068)
    @test isapprox(coef(gm20)[1:7],
        [2.894527697811509, -0.5693411448715979, 0.08238813087070128, -0.4484636623590206,
         0.08805060372902418, 0.3569553124412582, 0.2921383118842893])

    @testset "NegativeBinomial Parameter estimation" begin
        # Issue #302
        df = DataFrame(y = [1, 1, 0, 2, 3, 0, 0, 1, 1, 0, 2, 1, 3, 1, 1, 1, 4])
        for maxiter in [30, 50]
            try
                negbin(@formula(y ~ 1), df, maxiter = maxiter,
                    # set minstepfac to a very small value to avoid an ErrorException
                    # instead of a ConvergenceException
                    minstepfac=1e-20)
            catch err
                if err isa ConvergenceException
                    @test err.iters == maxiter
                else
                    rethrow(err)
                end
            end
        end
    end
end

@testset "NegativeBinomial NegativeBinomialLink, θ to be estimated" begin
    # the default/canonical link is NegativeBinomialLink
    gm21 = negbin(@formula(Days ~ Eth+Sex+Age+Lrn), quine)
    test_show(gm21)
    @test dof(gm21) == 8
    @test isapprox(deviance(gm21), 168.0465485656672, rtol = 1e-7)
    @test isapprox(loglikelihood(gm21), -546.8048603957335, rtol = 1e-7)
    @test isapprox(aic(gm21), 1109.609720791467)
    @test isapprox(aicc(gm21), 1110.660815681978)
    @test isapprox(bic(gm21), 1133.4785737651337)
    @test isapprox(coef(gm21)[1:7],
        [-0.08288628676491684, -0.03697387258037785, 0.010284124099280421, -0.027411445371127288,
         0.01582155341041012, 0.029074956147127032, 0.023628812427424876])
end

@testset "Sparse GLM" begin
    rng = StableRNG(1)
    X = sprand(rng, 1000, 10, 0.01)
    β = randn(rng, 10)
    y = Bool[rand(rng) < logistic(x) for x in X * β]
    gmsparse = fit(GeneralizedLinearModel, X, y, Binomial())
    gmdense = fit(GeneralizedLinearModel, Matrix(X), y, Binomial())

    @test isapprox(deviance(gmsparse), deviance(gmdense))
    @test isapprox(coef(gmsparse), coef(gmdense))
    @test isapprox(vcov(gmsparse), vcov(gmdense))
end

@testset "Predict" begin
    rng = StableRNG(123)
    X = rand(rng, 10, 2)
    Y = logistic.(X * [3; -3])

    gm11 = fit(GeneralizedLinearModel, X, Y, Binomial())
    @test isapprox(predict(gm11), Y)
    @test predict(gm11) == fitted(gm11)
    
    newX = rand(rng, 5, 2)
    newY = logistic.(newX * coef(gm11))
    gm11_pred1 = predict(gm11, newX)
    gm11_pred2 = predict(gm11, newX; interval=:confidence, interval_method=:delta)
    gm11_pred3 = predict(gm11, newX; interval=:confidence, interval_method=:transformation)
    @test gm11_pred1 == gm11_pred2.prediction == gm11_pred3.prediction≈ newY
    J = newX.*last.(GLM.inverselink.(LogitLink(), newX*coef(gm11)))
    se_pred = sqrt.(diag(J*vcov(gm11)*J'))
    @test gm11_pred2.lower ≈ gm11_pred2.prediction .- quantile(Normal(), 0.975).*se_pred ≈
        [0.20478201781547786, 0.2894172253195125, 0.17487705636545708, 0.024943206131575357, 0.41670326978944977]
    @test gm11_pred2.upper ≈ gm11_pred2.prediction .+ quantile(Normal(), 0.975).*se_pred ≈
        [0.6813754418027714, 0.9516561735593941, 1.0370309285468602, 0.5950732511233356, 1.192883895763427]


    off = rand(rng, 10)
    newoff = rand(rng, 5)

    @test_throws ArgumentError predict(gm11, newX, offset=newoff)

    gm12 = fit(GeneralizedLinearModel, X, Y, Binomial(), offset=off)
    @test_throws ArgumentError predict(gm12, newX)
    @test isapprox(predict(gm12, newX, offset=newoff),
        logistic.(newX * coef(gm12) .+ newoff))

    # Prediction from DataFrames
    d = DataFrame(X, :auto)
    d.y = Y

    gm13 = fit(GeneralizedLinearModel, @formula(y ~ 0 + x1 + x2), d, Binomial())
    @test predict(gm13) ≈ predict(gm13, d[:,[:x1, :x2]])
    @test predict(gm13) ≈ predict(gm13, d)

    newd = DataFrame(newX, :auto)
    predict(gm13, newd)

    Ylm = X * [0.8, 1.6] + 0.8randn(rng, 10)
    mm = fit(LinearModel, X, Ylm)
    pred1 = predict(mm, newX)
    pred2 = predict(mm, newX, interval=:confidence)
    se_pred = sqrt.(diag(newX*vcov(mm)*newX'))

    @test pred1 == pred2.prediction ≈
        [1.1382137814295972, 1.2097057044789292, 1.7983095679661645, 1.0139576473310072, 0.9738243263215998]
    @test pred2.lower ≈ pred2.prediction - quantile(TDist(dof_residual(mm)), 0.975)*se_pred ≈
        [0.5483482828723035, 0.3252331944785751, 0.6367574076909834, 0.34715818536935505, -0.41478974520958345]
    @test pred2.upper ≈ pred2.prediction + quantile(TDist(dof_residual(mm)), 0.975)*se_pred ≈
        [1.7280792799868907, 2.0941782144792835, 2.9598617282413455, 1.6807571092926594, 2.362438397852783]

    pred3 = predict(mm, newX, interval=:prediction)
    @test pred1 == pred3.prediction ≈
        [1.1382137814295972, 1.2097057044789292, 1.7983095679661645, 1.0139576473310072, 0.9738243263215998]
    @test pred3.lower ≈ pred3.prediction - quantile(TDist(dof_residual(mm)), 0.975)*sqrt.(diag(newX*vcov(mm)*newX') .+ deviance(mm)/dof_residual(mm)) ≈
        [-1.6524055967145255, -1.6576810549645142, -1.1662846080257512, -1.7939306570282658, -2.0868723667435027]
    @test pred3.upper ≈ pred3.prediction + quantile(TDist(dof_residual(mm)), 0.975)*sqrt.(diag(newX*vcov(mm)*newX') .+ deviance(mm)/dof_residual(mm)) ≈
        [3.9288331595737196, 4.077092463922373, 4.762903743958081, 3.82184595169028, 4.034521019386702]

    # Prediction with dropcollinear (#409)
    x = [1.0 1.0
         1.0 2.0
         1.0 -1.0]
    y = [1.0, 3.0, -2.0]
    m1 = lm(x, y, dropcollinear=true)
    m2 = lm(x, y, dropcollinear=false)

    p1 = predict(m1, x, interval=:confidence)
    p2 = predict(m2, x, interval=:confidence)

    @test p1.prediction ≈ p2.prediction
    @test p1.upper ≈ p2.upper
    @test p1.lower ≈ p2.lower

    # Prediction with dropcollinear and complex column permutations (#431)
    x = [1.0 100.0 1.2
         1.0 20000.0 2.3
         1.0 -1000.0 4.6
         1.0 5000 2.4]
    y = [1.0, 3.0, -2.0, 4.5]
    m1 = lm(x, y, dropcollinear=true)
    m2 = lm(x, y, dropcollinear=false)

    p1 = predict(m1, x, interval=:confidence)
    p2 = predict(m2, x, interval=:confidence)

    @test p1.prediction ≈ p2.prediction
    @test p1.upper ≈ p2.upper
    @test p1.lower ≈ p2.lower

    # Deprecated argument value
    @test predict(m1, x, interval=:confint) == p1

    # Prediction intervals would give incorrect results when some variables
    # have been dropped due to collinearity (#410)
    x = [1.0 1.0 2.0
         1.0 2.0 3.0
         1.0 -1.0 0.0]
    y = [1.0, 3.0, -2.0]
    m1 = lm(x, y)
    m2 = lm(x[:, 1:2], y)

    @test predict(m1) ≈ predict(m2)
    @test_broken predict(m1, interval=:confidence) ≈
        predict(m2, interval=:confidence)
    @test_broken predict(m1, interval=:prediction) ≈
        predict(m2, interval=:prediction)
    @test_throws ArgumentError predict(m1, x, interval=:confidence)
    @test_throws ArgumentError predict(m1, x, interval=:prediction)
end

@testset "GLM confidence intervals" begin
    X = [fill(1,50) range(0,1, length=50)]
    Y = vec([0 0 0 1 0 1 1 0 0 0 0 0 0 0 1 0 1 1 0 1 1 0 1 0 0 1 1 1 0 1 1 1 1 1 0 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1])
    gm = fit(GeneralizedLinearModel, X, Y, Binomial())

    newX = [fill(1,5) [0.0000000, 0.2405063, 0.4936709, 0.7468354, 1.0000000]]

    ggplot_prediction = [0.1804678, 0.3717731, 0.6262062, 0.8258605, 0.9306787]
    ggplot_lower = [0.05704968, 0.20624382, 0.46235427, 0.63065189, 0.73579237]
    ggplot_upper = [0.4449066, 0.5740713, 0.7654544, 0.9294403, 0.9847846]

    R_glm_se = [0.09748766, 0.09808412, 0.07963897, 0.07495792, 0.05177654]

    preds_transformation = predict(gm, newX, interval=:confidence, interval_method=:transformation)
    preds_delta = predict(gm, newX, interval=:confidence, interval_method=:delta)

    @test preds_transformation.prediction == preds_delta.prediction
    @test preds_transformation.prediction ≈ ggplot_prediction atol=1e-3
    @test preds_transformation.lower ≈ ggplot_lower atol=1e-3
    @test preds_transformation.upper ≈ ggplot_upper atol=1e-3

    @test preds_delta.upper .-  preds_delta.lower ≈ 2 .* 1.96 .* R_glm_se atol=1e-3
    @test_throws ArgumentError predict(gm, newX, interval=:confidence, interval_method=:undefined_method)
    @test_throws ArgumentError predict(gm, newX, interval=:undefined)
end

@testset "F test comparing to null model" begin
    d = DataFrame(Treatment=[1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2.],
                  Result=[1.1, 1.2, 1, 2.2, 1.9, 2, .9, 1, 1, 2.2, 2, 2],
                  Other=categorical([1, 1, 2, 1, 2, 1, 3, 1, 1, 2, 2, 1]))
    mod = lm(@formula(Result~Treatment), d).model
    othermod = lm(@formula(Result~Other), d).model
    nullmod = lm(@formula(Result~1), d).model
    bothmod = lm(@formula(Result~Other+Treatment), d).model
    nointerceptmod = lm(reshape(d.Treatment, :, 1), d.Result)

    ft1 = ftest(mod)
    ft1base = ftest(nullmod, mod)
    @test ft1.nobs == ft1base.nobs
    @test ft1.dof ≈ dof(mod) - dof(nullmod)
    @test ft1.fstat ≈ ft1base.fstat[2]
    @test ft1.pval ≈ ft1base.pval[2]
    if VERSION >= v"1.6.0"
        @test sprint(show, ft1) == """
            F-test against the null model:
            F-statistic: 241.62 on 12 observations and 1 degrees of freedom, p-value: <1e-07"""
    else
        @test sprint(show, ft1) == """
            F-test against the null model:
            F-statistic: 241.62 on 12 observations and 1 degrees of freedom, p-value: <1e-7"""
    end

    ft2 = ftest(othermod)
    ft2base = ftest(nullmod, othermod)
    @test ft2.nobs == ft2base.nobs
    @test ft2.dof ≈ dof(othermod) - dof(nullmod)
    @test ft2.fstat ≈ ft2base.fstat[2]
    @test ft2.pval ≈ ft2base.pval[2]
    @test sprint(show, ft2) == """
        F-test against the null model:
        F-statistic: 1.12 on 12 observations and 2 degrees of freedom, p-value: 0.3690"""

    ft3 = ftest(bothmod)
    ft3base = ftest(nullmod, bothmod)
    @test ft3.nobs == ft3base.nobs
    @test ft3.dof ≈ dof(bothmod) - dof(nullmod)
    @test ft3.fstat ≈ ft3base.fstat[2]
    @test ft3.pval ≈ ft3base.pval[2]
    if VERSION >= v"1.6.0"
        @test sprint(show, ft3) == """
            F-test against the null model:
            F-statistic: 81.97 on 12 observations and 3 degrees of freedom, p-value: <1e-05"""
    else
        @test sprint(show, ft3) == """
            F-test against the null model:
            F-statistic: 81.97 on 12 observations and 3 degrees of freedom, p-value: <1e-5"""
    end

    @test_throws ArgumentError ftest(nointerceptmod)
end

@testset "F test for model comparison" begin
    d = DataFrame(Treatment=[1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2.],
                  Result=[1.1, 1.2, 1, 2.2, 1.9, 2, .9, 1, 1, 2.2, 2, 2],
                  Other=categorical([1, 1, 2, 1, 2, 1, 3, 1, 1, 2, 2, 1]))
    mod = lm(@formula(Result~Treatment), d).model
    othermod = lm(@formula(Result~Other), d).model
    nullmod = lm(@formula(Result~1), d).model
    bothmod = lm(@formula(Result~Other+Treatment), d).model
    @test StatsModels.isnested(nullmod, mod)
    @test !StatsModels.isnested(othermod, mod)
    @test StatsModels.isnested(mod, bothmod)
    @test !StatsModels.isnested(bothmod, mod)
    @test StatsModels.isnested(othermod, bothmod)

    d.Sum = d.Treatment + (d.Other .== 1)
    summod = lm(@formula(Result~Sum), d).model
    @test StatsModels.isnested(summod, bothmod)

    ft1a = ftest(mod, nullmod)
    @test isnan(ft1a.pval[1])
    @test ft1a.pval[2] ≈ 2.481215056713184e-8
    if VERSION >= v"1.6.0"
        @test sprint(show, ft1a) == """
            F-test: 2 models fitted on 12 observations
            ──────────────────────────────────────────────────────────────────
                 DOF  ΔDOF     SSR    ΔSSR       R²      ΔR²        F*   p(>F)
            ──────────────────────────────────────────────────────────────────
            [1]    3        0.1283           0.9603                           
            [2]    2    -1  3.2292  3.1008  -0.0000  -0.9603  241.6234  <1e-07
            ──────────────────────────────────────────────────────────────────"""
    else
        @test sprint(show, ft1a) == """
            F-test: 2 models fitted on 12 observations
            ─────────────────────────────────────────────────────────────────
                 DOF  ΔDOF     SSR    ΔSSR       R²      ΔR²        F*  p(>F)
            ─────────────────────────────────────────────────────────────────
            [1]    3        0.1283           0.9603                          
            [2]    2    -1  3.2292  3.1008  -0.0000  -0.9603  241.6234  <1e-7
            ─────────────────────────────────────────────────────────────────"""
    end

    ft1b = ftest(nullmod, mod)
    @test isnan(ft1b.pval[1])
    @test ft1b.pval[2] ≈ 2.481215056713184e-8
    if VERSION >= v"1.6.0"
        @test sprint(show, ft1b) == """
            F-test: 2 models fitted on 12 observations
            ──────────────────────────────────────────────────────────────────
                 DOF  ΔDOF     SSR     ΔSSR       R²     ΔR²        F*   p(>F)
            ──────────────────────────────────────────────────────────────────
            [1]    2        3.2292           -0.0000                          
            [2]    3     1  0.1283  -3.1008   0.9603  0.9603  241.6234  <1e-07
            ──────────────────────────────────────────────────────────────────"""
    else
        @test sprint(show, ft1b) == """
            F-test: 2 models fitted on 12 observations
            ─────────────────────────────────────────────────────────────────
                 DOF  ΔDOF     SSR     ΔSSR       R²     ΔR²        F*  p(>F)
            ─────────────────────────────────────────────────────────────────
            [1]    2        3.2292           -0.0000                         
            [2]    3     1  0.1283  -3.1008   0.9603  0.9603  241.6234  <1e-7
            ─────────────────────────────────────────────────────────────────"""
    end

    bigmod = lm(@formula(Result~Treatment+Other), d).model
    ft2a = ftest(nullmod, mod, bigmod)
    @test isnan(ft2a.pval[1])
    @test ft2a.pval[2] ≈ 2.481215056713184e-8
    @test ft2a.pval[3] ≈ 0.3949973540194818
    if VERSION >= v"1.6.0"
        @test sprint(show, ft2a) == """
            F-test: 3 models fitted on 12 observations
            ──────────────────────────────────────────────────────────────────
                 DOF  ΔDOF     SSR     ΔSSR       R²     ΔR²        F*   p(>F)
            ──────────────────────────────────────────────────────────────────
            [1]    2        3.2292           -0.0000                          
            [2]    3     1  0.1283  -3.1008   0.9603  0.9603  241.6234  <1e-07
            [3]    5     2  0.1017  -0.0266   0.9685  0.0082    1.0456  0.3950
            ──────────────────────────────────────────────────────────────────"""
    else
        @test sprint(show, ft2a) == """
            F-test: 3 models fitted on 12 observations
            ──────────────────────────────────────────────────────────────────
                 DOF  ΔDOF     SSR     ΔSSR       R²     ΔR²        F*   p(>F)
            ──────────────────────────────────────────────────────────────────
            [1]    2        3.2292           -0.0000                          
            [2]    3     1  0.1283  -3.1008   0.9603  0.9603  241.6234   <1e-7
            [3]    5     2  0.1017  -0.0266   0.9685  0.0082    1.0456  0.3950
            ──────────────────────────────────────────────────────────────────"""
    end

    ft2b = ftest(bigmod, mod, nullmod)
    @test isnan(ft2b.pval[1])
    @test ft2b.pval[2] ≈ 0.3949973540194818
    @test ft2b.pval[3] ≈ 2.481215056713184e-8
    if VERSION >= v"1.6.0"
        @test sprint(show, ft2b) == """
            F-test: 3 models fitted on 12 observations
            ──────────────────────────────────────────────────────────────────
                 DOF  ΔDOF     SSR    ΔSSR       R²      ΔR²        F*   p(>F)
            ──────────────────────────────────────────────────────────────────
            [1]    5        0.1017           0.9685                           
            [2]    3    -2  0.1283  0.0266   0.9603  -0.0082    1.0456  0.3950
            [3]    2    -1  3.2292  3.1008  -0.0000  -0.9603  241.6234  <1e-07
            ──────────────────────────────────────────────────────────────────"""
    else
        @test sprint(show, ft2b) == """
            F-test: 3 models fitted on 12 observations
            ──────────────────────────────────────────────────────────────────
                 DOF  ΔDOF     SSR    ΔSSR       R²      ΔR²        F*   p(>F)
            ──────────────────────────────────────────────────────────────────
            [1]    5        0.1017           0.9685                           
            [2]    3    -2  0.1283  0.0266   0.9603  -0.0082    1.0456  0.3950
            [3]    2    -1  3.2292  3.1008  -0.0000  -0.9603  241.6234   <1e-7
            ──────────────────────────────────────────────────────────────────"""
    end

    @test_throws ArgumentError ftest(mod, bigmod, nullmod)
    @test_throws ArgumentError ftest(nullmod, bigmod, mod)
    @test_throws ArgumentError ftest(bigmod, nullmod, mod)
    mod2 = lm(@formula(Result~Treatment), d[2:end, :]).model
    @test_throws ArgumentError ftest(mod, mod2)
end

@testset "F test rounding error" begin
    # Data and Regressors
    Y = [8.95554, 10.7601, 11.6401, 6.53665, 9.49828, 10.5173, 9.34927, 5.95772, 6.87394, 9.56881, 13.0369, 10.1762]
    X = [1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0;
         0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 1.0 0.0 1.0 0.0]'
    # Correlation matrix
    V = [7.0  0.0  0.0    0.0      0.0    0.0    0.0    0.0      0.0      0.0    0.0      0.0
         0.0  7.0  0.0    0.0      0.0    0.0    0.0    0.0      0.0      0.0    0.0      3.0
         0.0  0.0  7.0    1.056    2.0    1.0    1.0    1.056    1.056    2.0    2.0      0.0
         0.0  0.0  1.056  6.68282  1.112  2.888  1.944  4.68282  5.68282  1.112  1.112    0.0
         0.0  0.0  2.0    1.112    7.0    1.0    1.0    1.112    1.112    5.0    4.004    0.0
         0.0  0.0  1.0    2.888    1.0    7.0    2.0    2.888    2.888    1.0    1.0      0.0
         0.0  0.0  1.0    1.944    1.0    2.0    7.0    1.944    1.944    1.0    1.0      0.0
         0.0  0.0  1.056  4.68282  1.112  2.888  1.944  6.68282  4.68282  1.112  1.112    0.0
         0.0  0.0  1.056  5.68282  1.112  2.888  1.944  4.68282  6.68282  1.112  1.112    0.0
         0.0  0.0  2.0    1.112    5.0    1.0    1.0    1.112    1.112    7.0    4.008    0.0
         0.0  0.0  2.0    1.112    4.004  1.0    1.0    1.112    1.112    4.008  6.99206  0.0
         0.0  3.0  0.0    0.0      0.0    0.0    0.0    0.0      0.0      0.0    0.0      7.0]
    # Cholesky
    RL = cholesky(V).L
    Yc = RL\Y
    # Fit 1 (intercept)
    Xc1 = RL\X[:,[1]]
    mod1 = lm(Xc1, Yc)
    # Fit 2 (both)
    Xc2 = RL\X
    mod2 = lm(Xc2, Yc)
    @test StatsModels.isnested(mod1, mod2)
end

@testset "coeftable" begin
    lm1 = fit(LinearModel, @formula(OptDen ~ Carb), form)
    t = coeftable(lm1)
    @test t.cols[1:3] ==
        [coef(lm1), stderror(lm1), coef(lm1)./stderror(lm1)]
    @test t.cols[4] ≈ [0.5515952883836446, 3.409192065429258e-7]
    @test hcat(t.cols[5:6]...) == confint(lm1)
    # TODO: call coeftable(gm1, ...) directly once DataFrameRegressionModel
    # supports keyword arguments
    t = coeftable(lm1.model, level=0.99)
    @test hcat(t.cols[5:6]...) == confint(lm1, level=0.99)

    gm1 = fit(GeneralizedLinearModel, @formula(Counts ~ 1 + Outcome + Treatment),
              dobson, Poisson())
    t = coeftable(gm1)
    @test t.cols[1:3] ==
        [coef(gm1), stderror(gm1), coef(gm1)./stderror(gm1)]
    @test t.cols[4] ≈ [5.4267674619082684e-71, 0.024647114627808674, 0.12848651178787643,
                       0.9999999999999981, 0.9999999999999999]
    @test hcat(t.cols[5:6]...) == confint(gm1)
    # TODO: call coeftable(gm1, ...) directly once DataFrameRegressionModel
    # supports keyword arguments
    t = coeftable(gm1.model, level=0.99)
    @test hcat(t.cols[5:6]...) == confint(gm1, level=0.99)
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

@testset "Issue 117" begin
    data = DataFrame(x = [1,2,3,4], y = [24,34,44,54])
    @test isapprox(coef(glm(@formula(y ~ x), data, Normal(), IdentityLink())), [14., 10])
end

@testset "Issue 118" begin
    @inferred nobs(lm(randn(10, 2), randn(10)))
end

@testset "Issue 153" begin
    X = [ones(10) randn(10)]
    Test.@inferred cholesky(GLM.DensePredQR{Float64}(X))
end

@testset "Issue 213" begin
    x, y, w = rand(100, 2), rand(100), rand(100)
    lm1 = lm(x, y)
    lm2 = lm(x, view(y, :))
    lm3 = lm(view(x, :, :), y)
    lm4 = lm(view(x, :, :), view(y, :))
    @test coef(lm1) ≈ coef(lm2) ≈ coef(lm3) ≈ coef(lm4)

    lm5 = lm(x, y, wts=w)
    lm6 = lm(x, view(y, :), wts=w)
    lm7 = lm(view(x, :, :), y, wts=w)
    lm8 = lm(view(x, :, :), view(y, :), wts=w)
    lm9 = lm(x, y, wts=view(w, :))
    lm10 = lm(x, view(y, :), wts=view(w, :))
    lm11 = lm(view(x, :, :), y, wts=view(w, :))
    lm12 = lm(view(x, :, :), view(y, :), wts=view(w, :))
    @test coef(lm5) ≈ coef(lm6) ≈ coef(lm7) ≈ coef(lm8) ≈ coef(lm9) ≈ coef(lm10) ≈ coef(lm11) ≈ coef(lm12)

    x, y, w = rand(100, 2), rand(Bool, 100), rand(100)
    glm1 = glm(x, y, Binomial())
    glm2 = glm(x, view(y, :), Binomial())
    @test coef(glm1) ≈ coef(glm2)

    glm3 = glm(x, y, Binomial(), wts=w)
    glm4 = glm(x, view(y, :), Binomial(), wts=w)
    glm5 = glm(x, y, Binomial(), wts=view(w, :))
    glm6 = glm(x, view(y, :), Binomial(), wts=view(w, :))
    @test coef(glm3) ≈ coef(glm4) ≈ coef(glm5) ≈ coef(glm6)
end

@testset "Issue 224" begin
    rng = StableRNG(1009)
    # Make X slightly ill conditioned to amplify rounding errors
    X = Matrix(qr(randn(rng, 100, 5)).Q)*Diagonal(10 .^ (-2.0:1.0:2.0))*Matrix(qr(randn(rng, 5, 5)).Q)'
    y = randn(rng, 100)
    @test coef(glm(X, y, Normal(), IdentityLink())) ≈ coef(lm(X, y))
end

@testset "Issue #228" begin
    @test_throws ArgumentError glm(randn(10, 2), rand(1:10, 10), Binomial(10))
end

@testset "Issue #263" begin
    data = dataset("datasets", "iris")
    data.SepalWidth2 = data.SepalWidth
    model1 = lm(@formula(SepalLength ~ SepalWidth), data)
    model2 = lm(@formula(SepalLength ~ SepalWidth + SepalWidth2), data, true)
    model3 = lm(@formula(SepalLength ~ 0 + SepalWidth), data)
    model4 = lm(@formula(SepalLength ~ 0 + SepalWidth + SepalWidth2), data, true)
    @test dof(model1) == dof(model2)
    @test dof(model3) == dof(model4)
    @test dof_residual(model1) == dof_residual(model2)
    @test dof_residual(model3) == dof_residual(model4)
end

@testset "Issue #286 (separable data)" begin
    x  = rand(1000)
    df = DataFrame(y = x .> 0.5, x₁ = x, x₂ = rand(1000))
    @testset "Binomial with $l" for l in (LogitLink(), ProbitLink(), CauchitLink(), CloglogLink())
        @test deviance(glm(@formula(y ~ x₁ + x₂), df, Binomial(), l, maxiter=40)) < 1e-6
    end
end

@testset "Issue #376 (== and isequal for links)" begin
    @test GLM.LogitLink() == GLM.LogitLink()
    @test NegativeBinomialLink(0.3) == NegativeBinomialLink(0.3)
    @test NegativeBinomialLink(0.31) != NegativeBinomialLink(0.3)

    @test isequal(GLM.LogitLink(), GLM.LogitLink())
    @test isequal(NegativeBinomialLink(0.3), NegativeBinomialLink(0.3))
    @test !isequal(NegativeBinomialLink(0.31), NegativeBinomialLink(0.3))

    @test hash(GLM.LogitLink()) == hash(GLM.LogitLink())
    @test hash(NegativeBinomialLink(0.3)) == hash(NegativeBinomialLink(0.3))
    @test hash(NegativeBinomialLink(0.31)) != hash(NegativeBinomialLink(0.3))
end

@testset "hasintercept" begin
    d = DataFrame(Treatment=[1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2.],
                  Result=[1.1, 1.2, 1, 2.2, 1.9, 2, .9, 1, 1, 2.2, 2, 2],
                  Other=categorical([1, 1, 2, 1, 2, 1, 3, 1, 1, 2, 2, 1]))

    mod = lm(@formula(Result~Treatment), d).model
    @test hasintercept(mod)

    nullmod = lm(@formula(Result~1), d).model
    @test hasintercept(nullmod)

    nointerceptmod = lm(reshape(d.Treatment, :, 1), d.Result)
    @test !hasintercept(nointerceptmod)

    rng = StableRNG(1234321)
    secondcolinterceptmod = glm([randn(rng, 5) ones(5)], ones(5), Binomial(), LogitLink())
    @test hasintercept(secondcolinterceptmod)
end
