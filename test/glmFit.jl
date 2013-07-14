using GLM

dobson = DataFrame(counts=[18.,17,15,20,10,20,25,13,12], outcome=gl(3,1,9), treatment=gl(3,3))
gm1 = fit(glm(:(counts ~ outcome + treatment), dobson, Poisson()));
println(gm1)
println("drsum(gm1) = $(sum(gm1.rr.devresid)) is called 'deviance' in R")

## Example from http://www.ats.ucla.edu/stat/r/dae/logit.htm
nm = download("http://www.ats.ucla.edu/stat/data/binary.csv", tempname())
df = within(readtable(nm),:(rank=compact(PooledDataArray(rank))))
rm(nm)
gm2 = glm(:(admit ~ gre + gpa + rank), df, Bernoulli())
println(gm2)
gm3 = glm(:(admit ~ gre + gpa + rank), df, Bernoulli(), ProbitLink());
println(gm3)
gm4 = glm(:(admit ~ gre + gpa + rank), df, Bernoulli(), CauchitLink());
println(gm4)
gm5 = glm(:(admit ~ gre + gpa + rank), df, Bernoulli(), CloglogLink());
println(gm5)
