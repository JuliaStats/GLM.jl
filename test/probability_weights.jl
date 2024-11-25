rng = StableRNG(123)
x1 = rand(rng, 50)
x2 = ifelse.(randn(rng, 50) .> 0, 1, 0)
y = ifelse.(0.004 .- 0.01 .* x1 .+ 1.5 .* x2 .+ randn(rng, 50) .> 0, 1, 0)
w = rand(rng, 50) * 6
w = floor.(w) .+ 1
df = DataFrame(y = y, x1 = x1, x2 = x2, w = w)
df.pweights = size(df, 1) .* (df.w ./ sum(df.w))

clotting = DataFrame(
    u = log.([5, 10, 15, 20, 30, 40, 60, 80, 100]),
    lot1 = [118, 58, 42, 35, 27, 25, 21, 19, 18],
    w = [1.5, 2.0, 1.1, 4.5, 2.4, 3.5, 5.6, 5.4, 6.7],
)

clotting.pweights = (clotting.w ./ sum(clotting.w))

quine = RDatasets.dataset("MASS", "quine")
quine.aweights = log.(3 .+ 3 .* quine.Days)
quine.pweights = size(quine, 1) .* (quine.aweights ./ sum(quine.aweights))

dobson = DataFrame(
    Counts = [18.0, 17, 15, 20, 10, 20, 25, 13, 12],
    Outcome = categorical(repeat(string.('A':'C'), outer = 3)),
    Treatment = categorical(repeat(string.('a':'c'), inner = 3)),
    w = [1, 2, 1, 2, 3, 4, 3, 2, 1],
)

dobson.pweights = size(dobson, 1) .* (dobson.w ./ sum(dobson.w))

@testset "Linear Model ftest/loglikelihod with $dmethod method" for dmethod ∈ (:cholesky, :qr)
    model_1 = lm(@formula(y ~ x1 + x2), df; wts=pweights(df.pweights), method = dmethod)
    X = hcat(ones(length(df.y)), df.x1, df.x2)
    model_2 = lm(X, y; wts=pweights(df.pweights))
    @test_throws ArgumentError ftest(model_1)
    @test_throws ArgumentError ftest(model_2) 
    @test_throws ArgumentError loglikelihood(model_1)
    @test_throws ArgumentError loglikelihood(model_2)
end

@testset "GLM: Binomial with LogitLink link - ProbabilityWeights with $dmethod method" for dmethod ∈ (:cholesky, :qr)
    model = glm(
        @formula(y ~ 1 + x1 + x2),
        df,
        Binomial(),
        LogitLink(),
        wts = pweights(df.pweights),
        method = dmethod,
        rtol = 1e-07,
    )
    @test_throws ArgumentError loglikelihood(model)
    @test deviance(model) ≈ 47.311214978934785 rtol = 1e-07
    @test nulldeviance(model) ≈ 60.82748267747685 rtol = 1e-07
    @test coef(model) ≈ [-0.5241460813701, 0.14468927249342, 2.487500063309] rtol = 1e-06
    @test dof_residual(model) == 47.0
    @test stderror(model) ≈ [1.07077535201799, 1.4966446912323, 0.7679252464101] rtol = 1e-05
end

@testset "GLM: Binomial with ProbitLink link - ProbabilityWeights with $dmethod method" for dmethod ∈ (:cholesky, :qr)
    model = glm(
        @formula(y ~ 1 + x1 + x2),
        df,
        Binomial(),
        ProbitLink(),
        wts = pweights(df.pweights),
        method = dmethod,
        rtol = 1e-09,
    )
    @test_throws ArgumentError loglikelihood(model)
    @test deviance(model) ≈ 47.280413566179 rtol = 1e-07
    @test nulldeviance(model) ≈ 60.82748267747685 rtol = 1e-07
    @test coef(model) ≈ [-0.379823362118, 0.17460125170132, 1.4927538978259] rtol = 1e-07
    @test dof_residual(model) == 47.0
    @test stderror(model) ≈ [0.6250657160317, 0.851366312489, 0.4423686640689] rtol = 1e-05
end

@testset "GLM: Binomial with CauchitLink link - ProbabilityWeights with $dmethod method" for dmethod ∈ (:cholesky, :qr)
    model = glm(
        @formula(y ~ 1 + x1 + x2),
        df,
        Binomial(),
        CauchitLink(),
        wts = pweights(df.pweights),
        method = dmethod,
        rtol = 1e-07,
    )
    @test_throws ArgumentError loglikelihood(model)
    @test deviance(model) ≈ 47.17915872474391 rtol = 1e-07
    @test nulldeviance(model) ≈ 60.82748267747685 rtol = 1e-07
    @test coef(model) ≈ [-0.007674579802284, -0.5378132620063, 2.994759904353] rtol = 1e-06
    @test dof_residual(model) == 47.0
    @test stderror(model) ≈ [1.020489214335, 1.5748610330014, 1.5057621596148] rtol = 1e-03
end

@testset "GLM: Binomial with CloglogLink link - ProbabilityWeights with $dmethod method" for dmethod ∈ (:cholesky, :qr)
    model = glm(
        @formula(y ~ 1 + x1 + x2),
        df,
        Binomial(),
        CloglogLink(),
        wts = pweights(df.pweights),
        method = dmethod,
        rtol = 1e-09,
    )
    @test_throws ArgumentError loglikelihood(model)
    @test deviance(model) ≈ 47.063354817529856 rtol = 1e-07
    @test nulldeviance(model) ≈ 60.82748267747685 rtol = 1e-07
    @test coef(model) ≈ [-0.9897210433718, 0.449902058467, 1.5467108410611] rtol = 1e-07
    ## Test broken because of https://github.com/JuliaStats/GLM.jl/issues/509
    @test dof_residual(model) == 47.0
    @test stderror(model) ≈ [0.647026270959, 0.74668663622095, 0.49056337945919] rtol = 1e-04
end

@testset "GLM: Gamma with LogLink link - ProbabilityWeights with $dmethod method" for dmethod ∈ (:cholesky, :qr)
    model = glm(
        @formula(lot1 ~ 1 + u),
        clotting,
        Gamma(),
        LogLink(),
        wts = pweights(clotting.pweights),
        method = dmethod,
        rtol = 1e-12,
        atol = 1e-9,
    )
    @test_throws ArgumentError loglikelihood(model)
    @test deviance(model) ≈ 0.012601328117859285 rtol = 1e-07
    @test nulldeviance(model) ≈ 0.28335799805430917 rtol = 1e-07
    @test coef(model) ≈ [5.325098274654255, -0.5495659110653159] rtol = 1e-5
    ## Test broken because of https://github.com/JuliaStats/GLM.jl/issues/509
    @test dof_residual(model) == 7.0
    @test stderror(model) ≈ [0.2651749940925478, 0.06706321966020713] rtol = 1e-07
end

@testset "GLM: NegativeBinomial(2) with LogLink link - ProbabilityWeights with $dmethod method" for dmethod ∈ (:cholesky, :qr)
    model = glm(
        @formula(Days ~ Eth + Sex + Age + Lrn),
        quine,
        NegativeBinomial(2),
        LogLink(),
        wts = pweights(quine.pweights),
        method = dmethod,
        atol = 1e-09,
    )
    @test_throws ArgumentError loglikelihood(model)
    @test deviance(model) ≈ 178.46174895746665 rtol = 1e-07
    @test nulldeviance(model) ≈ 214.52243528092782 rtol = 1e-07
    @test coef(model) ≈ [
        3.0241191551553044,
        -0.46415766516885565,
        0.07185609429925505,
        -0.47848540911607695,
        0.09677889908013788,
        0.3562972562034377,
        0.34801618219815034,
    ] rtol = 1e-04
    @test dof_residual(model) == 139.0
    @test_broken stderror(model) ≈ [
        0.20080246284436692,
        0.14068933863735536,
        0.1440710375321996,
        0.2533527583247213,
        0.2401168459633955,
        0.23210823521812646,
        0.19039099362430775,
    ] rtol = 1e-05
    @test stderror(model) ≈ [
        0.20080246284436692,
        0.14068933863735536,
        0.1440710375321996,
        0.2533527583247213,
        0.2401168459633955,
        0.23210823521812646,
        0.19039099362430775,
    ] rtol = 1e-04
end

@testset "GLM:  with LogLink link - ProbabilityWeights with $dmethod method" for dmethod ∈ (:cholesky, :qr)
    model = glm(
        @formula(Days ~ Eth + Sex + Age + Lrn),
        quine,
        NegativeBinomial(2),
        LogLink(),
        wts = pweights(quine.pweights),
        method = dmethod,
        rtol = 1e-09,
    )
    @test_throws ArgumentError loglikelihood(model)
    @test deviance(model) ≈ 178.46174895746665 rtol = 1e-07
    @test nulldeviance(model) ≈ 214.52243528092782 rtol = 1e-07
    @test coef(model) ≈ [
        3.0241191551553044,
        -0.46415766516885565,
        0.07185609429925505,
        -0.47848540911607695,
        0.09677889908013788,
        0.3562972562034377,
        0.34801618219815034,
    ] rtol = 1e-04
    @test dof_residual(model) == 139.0
    @test stderror(model) ≈ [
        0.20080246284436692,
        0.14068933863735536,
        0.1440710375321996,
        0.2533527583247213,
        0.2401168459633955,
        0.23210823521812646,
        0.19039099362430775,
    ] rtol = 1e-04
end

@testset "GLM: NegaiveBinomial(2) with SqrtLink link - ProbabilityWeights with $dmethod method" for dmethod ∈ (:cholesky, :qr)
    model = glm(
        @formula(Days ~ Eth + Sex + Age + Lrn),
        quine,
        NegativeBinomial(2),
        SqrtLink(),
        wts = pweights(quine.pweights),
        method = dmethod,
        rtol = 1e-08,
    )
    @test_throws ArgumentError loglikelihood(model)
    @test deviance(model) ≈ 178.99970038364276 rtol = 1e-07
    @test nulldeviance(model) ≈ 214.52243528092782 rtol = 1e-07
    @test coef(model) ≈ [
        4.733877229152367,
        -1.0079778954713488,
        0.025223928185488836,
        -0.985974316804644,
        0.2132095063819702,
        0.7456070470961171,
        0.5840284357554048,
    ] rtol = 1e-07
    
    @test dof_residual(model) == 139.0
    @test stderror(model) ≈ [
        0.4156607040373307,
        0.30174203746555045,
        0.30609799754882105,
        0.526030598769091,
        0.5384102946567921,
        0.5328456049279787,
        0.4065359817407846,
    ] rtol = 1e-04
end

@testset "GLM: Poisson with LogLink link - ProbabilityWeights with $dmethod method" for dmethod ∈ (:cholesky, :qr)
    model = glm(
        @formula(Counts ~ 1 + Outcome + Treatment),
        dobson,
        Poisson(),
        LogLink(),
        wts = pweights(dobson.pweights),
        method = dmethod,
    )
    @test_throws ArgumentError loglikelihood(model)
    @test deviance(model) ≈ 4.837327189925912 rtol = 1e-07
    @test nulldeviance(model) ≈ 12.722836814903907 rtol = 1e-07
    @test coef(model) ≈ [
        3.1097109912423444,
        -0.5376892683400354,
        -0.19731134600684794,
        -0.05011966661241072,
        0.010415729161988225,
    ] rtol = 1e-07    
    @test dof_residual(model) == 4.0
    @test stderror(model) ≈ [
        0.15474638805584298,
        0.13467582259453692,
        0.1482320418486368,
        0.17141304156534284,
        0.17488650070332398,
    ] rtol = 1e-06
end
