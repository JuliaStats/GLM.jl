using GLM, DataFrames, Compat
glm(y ~ 1, DataFrame(y = float64(bitrand(10))), Binomial())

n = 2_500_000; srand(1234321)
df2 = DataFrame(x1 = rand(Normal(), n),
                x2 = rand(Exponential(), n),
                ss = pool(rand(DiscreteUniform(50), n)));
mf = ModelFrame(x1 ~ x1 + x2 + ss, df2)
mm = ModelMatrix(mf)
beta = unshift!(rand(Normal(),52), 0.5); # "true" parameter values

## Create linear predictor and mean response
eta = mm.m * beta;
mu = [linkinv(LogitLink(), x) for x in eta]
y = map(Float64, rand(n) .< mu);        # simulate observed responses

gm6 = glm(mm.m, y, Binomial())
@time glm(mm.m, y, Binomial());
# Profile.print()
