using DataFrames, Distributions, GLM

df = DataFrame(quote
    counts  = [18.,17,15,20,10,20,25,13,12]
    outcome = PooledDataArray([1, 2, 3, 1, 2, 3, 1, 2, 3])
    treatment = PooledDataArray([1, 1, 1, 2, 2, 2, 3, 3, 3])
end)

glm(:(counts ~ outcome + treatment), df, Poisson())

srand(1234321)
mu = rand(1000)
y  = [rand() < m ? 1. : 0. for m in mu]
rr = GlmResp(Bernoulli(), y)
pp = DensePredQR(ones(Float64, (1000,1)))
glmfit(pp, rr)
pp = DensePredChol(ones(Float64, (1000,1)))
rr = GlmResp(Bernoulli(), y)
glmfit(pp, rr)
println("Estimates: $(pp.beta')")
println("Deviance:  $(deviance(rr))")

## A large simulated example
mu = rand(1_000_000)
rr = GlmResp(Bernoulli(), [rand() < m ? 1. : 0. for m in mu])
pp = DensePredChol(hcat(ones(Float64, size(mu)), randn(size(mu,1), 39)))
glmfit(pp, rr)
## redo for timing
pp.beta[:] = 0.
rr = GlmResp(Bernoulli(), rr.y)
println(@elapsed glmfit(pp,rr))

## Example from Dobson (1990), Page 93, Randomized Clinical Trial
rr = GlmResp(Poisson(), [18.,17,15,20,10,20,25,13,12])
ct3 = contr_treatment(3)
pp = DensePredQR(hcat(ones(Float64,9), indicators(gl(3,1,9))[1]*ct3, indicators(gl(3,3))[1]*ct3))
glmfit(pp, rr)
println("Estimates: $(pp.beta')")
println("Deviance:  $(deviance(rr))")

## Example from http://www.ats.ucla.edu/stat/r/dae/logit.htm
## First download http://www.ats.ucla.edu/stat/data/binary.csv and delete the first line
dd = readcsv("./binary.csv")
pp = DensePredQR(hcat(ones(Float64,size(dd,1)), dd[:,2:3], indicators(dd[:,4])[1]*contr_treatment(4)))
rr = GlmResp(Bernoulli(), dd[:,1])
glmfit(pp,rr)
println("Estimates: $(pp.beta')")
println("Deviance:  $(deviance(rr))")
