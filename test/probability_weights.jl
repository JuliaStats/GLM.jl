rng = StableRNG(123)
x1 = rand(rng, 50)
x2 = ifelse.(randn(rng, 50) .> 0, 1, 0)
y = ifelse.(0.004 .- 0.01 .* x1 .+ 1.5 .* x2 .+ randn(rng, 50) .> 0, 1, 0)
df = DataFrame(; y=y, x1=x1, x2=x2, pweights=floor.(rand(rng, 50) * 6) .+ 1)

clotting = DataFrame(; u=log.([5, 10, 15, 20, 30, 40, 60, 80, 100]),
                     lot1=[118, 58, 42, 35, 27, 25, 21, 19, 18],
                     pweights=[1.5, 2.0, 1.1, 4.5, 2.4, 3.5, 5.6, 5.4, 6.7])

quine = RDatasets.dataset("MASS", "quine")
quine.pweights = log.(3 .+ 3 .* quine.Days)
dobson = DataFrame(; Counts=[18.0, 17, 15, 20, 10, 20, 25, 13, 12],
                   Outcome=categorical(repeat(string.('A':'C'); outer=3)),
                   Treatment=categorical(repeat(string.('a':'c'); inner=3)),
                   pweights=[1, 2, 1, 2, 3, 4, 3, 2, 1])

itr = Iterators.product((:qr, :cholesky), (true, false))

@testset "Linear Model ftest/loglikelihod with $dmethod method with dropcollinear=$drop" for (dmethod,
                                                                                              drop) in
                                                                                             itr

    model_1 = lm(@formula(y ~ x1 + x2), df; wts=pweights(df.pweights), method=dmethod)
    X = hcat(ones(length(df.y)), df.x1, df.x2)
    model_2 = lm(X, y; wts=pweights(df.pweights))
    @test_throws ArgumentError ftest(model_1)
    @test_throws ArgumentError ftest(model_2)
    @test_throws ArgumentError loglikelihood(model_1)
    @test_throws ArgumentError loglikelihood(model_2)
end

@testset "GLM: Binomial with LogitLink link - ProbabilityWeights with $dmethod method with dropcollinear=$drop" for (dmethod,
                                                                                                                     drop) in
                                                                                                                    itr

    model = glm(@formula(y ~ 1 + x1 + x2),
                df,
                Binomial(),
                LogitLink();
                wts=pweights(df.pweights),
                method=dmethod,
                dropcollinear=drop,
                rtol=1e-09,
                atol=1e-09)
    @test_throws ArgumentError loglikelihood(model)
    @test deviance(model) ≈ 47.311214978934785 rtol = 1e-07
    @test nulldeviance(model) ≈ 60.82748267747685 rtol = 1e-07
    @test coef(model) ≈ [-0.5241460813701, 0.14468927249342, 2.487500063309] rtol = 1e-06
    @test dof_residual(model) == 47.0
    @test stderror(model) ≈ [1.07077535201799, 1.4966446912323, 0.7679252464101] rtol = 1e-05
end

@testset "GLM: Binomial with ProbitLink link - ProbabilityWeights with $dmethod method  with dropcollinear=$drop" for (dmethod,
                                                                                                                       drop) in
                                                                                                                      itr

    model = glm(@formula(y ~ 1 + x1 + x2),
                df,
                Binomial(),
                ProbitLink();
                wts=pweights(df.pweights),
                method=dmethod,
                dropcollinear=drop,
                rtol=1e-09,
                atol=1e-09)
    @test_throws ArgumentError loglikelihood(model)
    @test deviance(model) ≈ 47.280413566179 rtol = 1e-07
    @test nulldeviance(model) ≈ 60.82748267747685 rtol = 1e-07
    @test coef(model) ≈ [-0.379823362118, 0.17460125170132, 1.4927538978259] rtol = 1e-05
    @test dof_residual(model) == 47.0
    @test stderror(model) ≈ [0.6250657160317, 0.851366312489, 0.4423686640689] rtol = 1e-05
end

@testset "GLM: Binomial with CauchitLink link - ProbabilityWeights with $dmethod method with dropcollinear=$drop" for (dmethod,
                                                                                                                       drop) in
                                                                                                                      itr

    model = glm(@formula(y ~ 1 + x1 + x2),
                df,
                Binomial(),
                CauchitLink();
                wts=pweights(df.pweights),
                method=dmethod,
                dropcollinear=drop,
                rtol=1e-09,
                atol=1e-09)
    @test_throws ArgumentError loglikelihood(model)
    @test deviance(model) ≈ 47.17915872474391 rtol = 1e-07
    @test nulldeviance(model) ≈ 60.82748267747685 rtol = 1e-07
    @test coef(model) ≈ [-0.007674579802284, -0.5378132620063, 2.994759904353] rtol = 1e-04
    @test dof_residual(model) == 47.0
    @test stderror(model) ≈ [1.020489214335, 1.5748610330014, 1.5057621596148] rtol = 1e-03
end

@testset "GLM: Binomial with CloglogLink link - ProbabilityWeights with $dmethod method with dropcollinear=$drop" for (dmethod,
                                                                                                                       drop) in
                                                                                                                      itr

    model = glm(@formula(y ~ 1 + x1 + x2),
                df,
                Binomial(),
                CloglogLink();
                wts=pweights(df.pweights),
                method=dmethod,
                dropcollinear=drop,
                rtol=1e-09,
                atol=1e-09)
    @test_throws ArgumentError loglikelihood(model)
    @test deviance(model) ≈ 47.063354817529856 rtol = 1e-07
    @test nulldeviance(model) ≈ 60.82748267747685 rtol = 1e-07
    @test coef(model) ≈ [-0.9897210433718, 0.449902058467, 1.5467108410611] rtol = 1e-04
    @test dof_residual(model) == 47.0
    @test stderror(model) ≈ [0.647026270959, 0.74668663622095, 0.49056337945919] rtol = 1e-04
end

@testset "GLM: Gamma with LogLink link - ProbabilityWeights with $dmethod method with dropcollinear=$drop" for (dmethod,
                                                                                                                drop) in
                                                                                                               itr

    model = glm(@formula(lot1 ~ 1 + u),
                clotting,
                Gamma(),
                LogLink();
                wts=pweights(clotting.pweights),
                method=dmethod,
                dropcollinear=drop,
                rtol=1e-9,
                atol=1e-9,
                minstepfac=1e-05)
    @test_throws ArgumentError loglikelihood(model)
    @test deviance(model) ≈ 0.113412 rtol = 1e-06
    @test nulldeviance(model) ≈ 2.55 rtol = 1e-04
    @test coef(model) ≈ [5.32511, -0.549568] rtol = 1e-6
    @test dof_residual(model) == 7.0
    @test stderror(model) ≈ [0.265172, 0.0670627] rtol = 1e-05
end

@testset "GLM: NegativeBinomial(2) with LogLink link - ProbabilityWeights with $dmethod method with dropcollinear=$drop" for (dmethod,
                                                                                                                              drop) in
                                                                                                                             itr

    model = glm(@formula(Days ~ Eth + Sex + Age + Lrn),
                quine,
                NegativeBinomial(2),
                LogLink();
                wts=pweights(quine.pweights),
                method=dmethod,
                dropcollinear=drop,
                atol=1e-09,
                rtol=1e-09,
                minstepfac=1e-04)
    @test_throws ArgumentError loglikelihood(model)
    @test deviance(model) ≈ 178.46174895746665 rtol = 1e-07
    @test nulldeviance(model) ≈ 214.52243528092782 rtol = 1e-07
    @test coef(model) ≈ [3.0241,
                         -0.464139,
                         0.07186,
                         -0.4785,
                         0.0967725,
                         0.356318,
                         0.348026] rtol = 1e-04
    @test dof_residual(model) == 139.0
    @test stderror(model) ≈ [0.20080,
                             0.14069,
                             0.14407,
                             0.25335,
                             0.24012,
                             0.23211,
                             0.19039] rtol = 1e-04
end

@testset "GLM:  NegativeBinomial(1) with LogLink link - ProbabilityWeights with $dmethod method with dropcollinear=$drop" for (dmethod,
                                                                                                                               drop) in
                                                                                                                              itr

    model = glm(@formula(Days ~ Eth + Sex + Age + Lrn),
                quine,
                NegativeBinomial(1),
                LogLink();
                wts=pweights(quine.pweights),
                method=dmethod,
                dropcollinear=drop,
                atol=1e-09,
                rtol=1e-09,
                minstepfac=1e-04)
    ## Geometric Link is NegativeBinomial(1)
    model_geom = glm(@formula(Days ~ Eth + Sex + Age + Lrn),
                     quine,
                     Geometric(),
                     LogLink();
                     wts=pweights(quine.pweights),
                     method=dmethod,
                     dropcollinear=drop,
                     atol=1e-09,
                     rtol=1e-09,
                     minstepfac=1e-04)
    @test_throws ArgumentError loglikelihood(model)
    @test_throws ArgumentError loglikelihood(model_geom)
    @test deviance(model) ≈ 98.45804 rtol = 1e-05
    @test nulldeviance(model) ≈ 117.407 rtol = 1e-05
    @test deviance(model_geom) ≈ 98.45804 rtol = 1e-05
    @test nulldeviance(model_geom) ≈ 117.407 rtol = 1e-05
    @test coef(model) ≈ [3.0312469487958,
                         -0.4659209078765,
                         0.0676685535488,
                         -0.4817223025756,
                         0.0931703051304,
                         0.3543515249482,
                         0.3437194303582] rtol = 1e-04
    @test coef(model_geom) ≈ coef(model) rtol = 1e-07
    @test dof_residual(model) == 139.0
    @test dof_residual(model) == dof_residual(model_geom)
    @test stderror(model) ≈ [0.20007959342369136,
                             0.14082290024757069,
                             0.14407240634114291,
                             0.25339272642416644,
                             0.2402419933610615,
                             0.23248097541210141,
                             0.19122111799256292] rtol = 1e-04
    @test stderror(model) ≈ stderror(model_geom) rtol = 1e-06
end

@testset "GLM: NegaiveBinomial(2) with SqrtLink link - ProbabilityWeights with $dmethod method with dropcollinear=$drop" for (dmethod,
                                                                                                                              drop) in
                                                                                                                             itr

    model = glm(@formula(Days ~ Eth + Sex + Age + Lrn),
                quine,
                NegativeBinomial(2),
                SqrtLink();
                wts=pweights(quine.pweights),
                method=dmethod,
                dropcollinear=drop,
                rtol=1e-08,
                atol=1e-08,
                minstepfac=1e-04)
    @test_throws ArgumentError loglikelihood(model)
    @test deviance(model) ≈ 178.99970038364276 rtol = 1e-07
    @test nulldeviance(model) ≈ 214.52243528092782 rtol = 1e-07
    @test coef(model) ≈ [4.733877229152367,
                         -1.0079778954713488,
                         0.025223928185488836,
                         -0.985974316804644,
                         0.2132095063819702,
                         0.7456070470961171,
                         0.5840284357554048] rtol = 1e-07

    @test dof_residual(model) == 139.0
    @test stderror(model) ≈ [0.4156607040373307,
                             0.30174203746555045,
                             0.30609799754882105,
                             0.526030598769091,
                             0.5384102946567921,
                             0.5328456049279787,
                             0.4065359817407846] rtol = 1e-04
end

@testset "GLM: Poisson with LogLink link - ProbabilityWeights with $dmethod method with dropcollinear=$drop" for (dmethod,
                                                                                                                  drop) in
                                                                                                                 itr

    model = glm(@formula(Counts ~ 1 + Outcome + Treatment),
                dobson,
                Poisson(),
                LogLink();
                wts=pweights(dobson.pweights),
                method=dmethod,
                dropcollinear=drop)
    @test_throws ArgumentError loglikelihood(model)
    @test deviance(model) ≈ 4.837327189925912 rtol = 1e-07
    @test nulldeviance(model) ≈ 12.722836814903907 rtol = 1e-07
    @test coef(model) ≈ [3.1097109912423444,
                         -0.5376892683400354,
                         -0.19731134600684794,
                         -0.05011966661241072,
                         0.010415729161988225] rtol = 1e-07
    @test dof_residual(model) == 4.0
    @test stderror(model) ≈ [0.15474638805584298,
                             0.13467582259453692,
                             0.1482320418486368,
                             0.17141304156534284,
                             0.17488650070332398] rtol = 1e-06
end

@testset "InverseGaussian ProbabilityWeights with $dmethod" for dmethod in (:cholesky, :qr)
    gm8a = fit(GeneralizedLinearModel, @formula(lot1 ~ 1 + u), clotting, InverseGaussian();
               wts=pweights(9 * clotting.pweights / sum(clotting.pweights)), method=dmethod,
               rtol=1e-08, atol=1e-08, minstepfac=1e-04,)
    @test dof(gm8a) == 3
    @test deviance(gm8a) ≈ 0.0058836 rtol = 1e-04
    @test nulldeviance(gm8a) ≈ 0.07531257 rtol = 1e-04
    @test coef(gm8a) ≈ [-0.001263439, 0.0008126671] rtol = 1e-04
    @test stderror(gm8a) ≈ [0.0001246, 7.655616888887675e-5] atol = 1e-06
end

@testset "Test sparse LM with ProbabilityWeights" for dmethod in (:cholesky, :qr)
    # Test sparse with probability weights
    rng = StableRNG(123)
    X = sprand(rng, 20, 10, 0.2)
    β = rand(rng, 10)
    y = X * β .+ randn(20)
    wts = rand(20)
    model_sparse = lm(X, y; wts=pweights(wts), method=dmethod)
    model_dense = lm(Matrix(X), y; wts=pweights(wts), method=dmethod)
    @test deviance(model_sparse) ≈ deviance(model_dense) rtol = 1e-07
    @test nulldeviance(model_sparse) ≈ nulldeviance(model_dense) rtol = 1e-07
    @test coef(model_sparse) ≈ coef(model_dense) rtol = 1e-07
    @test stderror(model_sparse) ≈ stderror(model_dense) rtol = 1e-07
    @test dof_residual(model_sparse) ≈ dof_residual(model_dense) rtol = 1e-07
    @test dof(model_sparse) ≈ dof(model_dense) rtol = 1e-07
end

@testset "Test sparse Rank-deficient LM ProbabilityWeights" begin
    # Test sparse with probability weights
    rng = StableRNG(123)
    X = sprand(rng, 20, 10, 0.2)
    β = rand(rng, 10)
    y = X * β .+ randn(20)
    X = hcat(X[:, 1:7], X[:, 1:2], X[:, 8:9], X[:, 6], X[:, 10]) # make it rank deficient
    wts = rand(20)
    model_sparse = lm(X, y; wts=pweights(wts), method=:qr)
    model_dense = lm(Matrix(X), y; wts=pweights(wts), method=:qr)
    @test deviance(model_sparse) ≈ deviance(model_dense) rtol = 1e-07
    @test nulldeviance(model_sparse) ≈ nulldeviance(model_dense) rtol = 1e-07
    se_sparse = stderror(model_sparse)
    se_dense = stderror(model_dense)
    isnan_sparse = isnan.(se_sparse)
    isnan_dense = isnan.(se_dense)
    @test sort(coef(model_sparse)[.!isnan_sparse]) ≈ sort(coef(model_dense)[.!isnan_dense])
    @test sort(se_sparse[.!isnan_sparse]) ≈ sort(se_dense[.!isnan_dense]) rtol = 1e-07
end
