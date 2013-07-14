using GLM

n = 2_500_000; srand(1234321)
df2 = DataFrame(x1 = rand(Normal(), n), x2 = rand(Exponential(), n),
                ss = pool(rand(DiscreteUniform(50), n)))
beta = unshift!(rand(Normal(),52), 0.5);
eta = ModelMatrix(ModelFrame(Formula(:(~ (x1 + x2 + ss))), df2)).m * beta
mu = similar(eta); linkinv!(LogitLink(), mu, eta)
df2["y"] = float64(rand(n) .< mu); df2["eta"] = eta; df2["mu"] = mu
@time gm6 = glm(:(y ~ x1 + x2 + ss), df2, Bernoulli())
