if false                                # need DataFrame package and formulas
    df = DataFrame(quote
        y  = [18.,17,15,20,10,20,25,13,12]
        x1 = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        x2 = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    end)

    glm(:(y ~ x1 + x2), df, Poisson())
end

using Distributions
using Glm
srand(1234321)
mu = rand(1000)
y  = [rand() < m ? 1. : 0. for m in mu]
rr = GlmResp(Bernoulli(), y)
pp = DensePredQR(ones(Float64, (1000,1)))
glmfit(pp, rr)
pp = DensePredChol(ones(Float64, (1000,1)))
rr = GlmResp(Bernoulli(), y)
glmfit(pp, rr)

y  = [18.,17,15,20,10,20,25,13,12]
x1 = [1, 2, 3, 1, 2, 3, 1, 2, 3]
x2 = [1, 1, 1, 2, 2, 2, 3, 3, 3]

rr = GlmResp(Poisson(), y)
ct3 = contr_treatment(3)
pp = DensePredQR(hcat(ones(Float64,9), indicators(gl(3,1,9))[1]*ct3, indicators(gl(3,3))[1]*ct3))
glmfit(pp, rr)
