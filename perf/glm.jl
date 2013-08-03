using GLM
glm(:(y ~ 1), DataFrame(y = float64(randbool(10))), Binomial())

n = 2_500_000; srand(1234321)
df2 = DataFrame(x1 = rand(Normal(), n),
                x2 = rand(Exponential(), n),
                ss = pool(rand(DiscreteUniform(50), n)));
beta = unshift!(rand(Normal(),52), 0.5);
eta = ModelMatrix(ModelFrame(Formula(:(~ (x1 + x2 + ss))), df2)).m * beta;
mu = linkinv!(LogitLink(), similar(eta), eta);
df2["y"] = float64(rand(n) .< mu); df2["eta"] = eta; df2["mu"] = mu;
head(df2)
gc()
gc()
@time gm6 = glm(:(y ~ x1 + x2 + ss), df2, Binomial(), LogitLink())
