using GLM, Random, StatsModels
                                # create a column table with dummy response
n = 2_500_000
rng = MersenneTwister(1234321)
tbl = (
    x1 = randn(rng, n),
    x2 = Random.randexp(rng, n),
    ss = rand(rng, string.(50:99), n),
    y = zeros(n),
)
                                # apply a formula to create a model matrix
f = @formula(y ~ 1 + x1 + x2 + ss)
f = apply_schema(f, schema(f, tbl))
resp, pred = modelcols(f, tbl)
                                # simulate β and the response
β = randn(rng, size(pred, 2))
β[1] = 0.5        # to avoid edge cases
logistic(x::Real) = inv(1 + exp(-x))
resp .= rand(rng, n) .< logistic.(pred * β)
                                # fit a subset of the data
gm6 = glm(pred[1:1000, :], resp[1:1000], Bernoulli())
                                # time the fit on the whole data set
@time glm(pred, resp, Bernoulli());
