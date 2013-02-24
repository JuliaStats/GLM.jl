using DataFrames, Distributions, GLM

dobson = DataFrame({[18.,17,15,20,10,20,25,13,12], gl(3,1,9), gl(3,3)},
                   ["counts","outcome","treatment"])
gm1 = glm(:(counts ~ outcome + treatment), dobson, Poisson())
deviance(gm1)                           # something wrong here

srand(1234321)
mu = rand(1000)
y  = [rand() < m ? 1. : 0. for m in mu]
rr = GlmResp(Bernoulli(), y)
pp = DensePredQR(ones(Float64, (1000,1)))
glmfit(pp, rr)
pp = DensePredChol(ones(Float64, (1000,1)))
rr = GlmResp(Bernoulli(), y)
glmfit(pp, rr)
typeof(pp).names
println("Estimates: $(pp.beta0')")
println("Deviance:  $(deviance(rr))")

## A large simulated example
mu = rand(1_000_000)
rr = GlmResp(Bernoulli(), [rand() < m ? 1. : 0. for m in mu])
pp = DensePredChol(hcat(ones(Float64, size(mu)), randn(size(mu,1), 39)))
glmfit(pp, rr)
## redo for timing
pp.beta0[:] = 0.
rr = GlmResp(Bernoulli(), rr.y)
println(@elapsed glmfit(pp,rr))


## Example from http://www.ats.ucla.edu/stat/r/dae/logit.htm
## First download http://www.ats.ucla.edu/stat/data/binary.csv and delete the first line
nm = download("http://www.ats.ucla.edu/stat/data/binary.csv", "/tmp/binary.csv")
dd = read_table(nm)
dd = within(dd, :(rank = PooledDataArray(rank)))
gm2 = glm(:(admit ~ gre + gpa + rank), dd, Bernoulli())
