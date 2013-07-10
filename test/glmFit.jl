using GLM

dobson = DataFrame({[18.,17,15,20,10,20,25,13,12], gl(3,1,9), gl(3,3)},
                   ["counts","outcome","treatment"])
gm1 = glm(:(counts ~ outcome + treatment), dobson, Poisson())
deviance(gm1)                           # something wrong here

## Example from http://www.ats.ucla.edu/stat/r/dae/logit.htm
nm = download("http://www.ats.ucla.edu/stat/data/binary.csv", tempname())
df = within(readtable(nm),:(rank=compact(PooledDataArray(rank))))
rm(nm)
gm2 = glm(:(admit ~ gre + gpa + rank), df, Bernoulli());
