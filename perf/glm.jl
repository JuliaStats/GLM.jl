using GLM, DataFrames, StatsFuns
glm(@formula(y ~ 1), DataFrame(y = float(bitrand(10))), Bernoulli())

n = 2_500_000
Random.seed!(1234321)
const df2 = DataFrame(x1 = rand(Normal(), n),
                x2 = rand(Exponential(), n),
                ss = pool(rand(DiscreteUniform(50), n)));
const mf = ModelFrame(@formula(x1 ~ x1 + x2 + ss), df2)
const mm = ModelMatrix(mf)
const β = pushfirst!(rand(Normal(),52), 0.5); # "true" parameter values

## Create linear predictor and mean response

const y = [float(rand() < logistic(η)) for η in mm.m * β];        # simulate observed responses

gm6 = glm(mm.m, y, Bernoulli())
@time glm(mm.m, y, Bernoulli());
#@profile glm(mm.m, y, Binomial());
#using ProfileView
#ProfileView.view()
