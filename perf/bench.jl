using PerfChecker
using BenchmarkTools

using GLM
using Random
using StatsModels

target = GLM

function bench()
    n = 25_00_000
    rng = Random.MersenneTwister(1234321)
    tbl = (
           x1 = randn(rng, n),
           x2 = Random.randexp(rng, n),
           ss = rand(rng, string.(50:99), n),
           y = zeros(n),
          )
    f = @formula(y ~ 1 + x1 + x2 + ss)
    f = apply_schema(f, schema(f, tbl))
    resp, pred = modelcols(f, tbl)
    B = randn(rng, size(pred, 2))
    B[1] = 0.5
    logistic(x::Real) = inv(1 + exp(-x))
    resp .= rand(rng, n) .< logistic.(pred * B)
    glm(pred, resp, Bernoulli())
    return nothing
end

t = @benchmark bench() evals = 1 samples = 100 seconds = 1000
store_benchmark(t, target; path=@__DIR__)
