using GLM

dobson = DataFrame({[18.,17,15,20,10,20,25,13,12], gl(3,1,9), gl(3,3)},
                   ["counts","outcome","treatment"])
gm1 = glm(:(counts ~ outcome + treatment), dobson, Poisson())
deviance(gm1)                           # something wrong here

## Example from http://www.ats.ucla.edu/stat/r/dae/logit.htm
nm = download("http://www.ats.ucla.edu/stat/data/binary.csv", tempname())
mm = float(readdlm(nm,',')[2:end,:])
df = DataFrame(admit=mm[:,1], gre=mm[:,2], gpa=mm[:,3], rank=PooledDataArray(mm[:,4]))
gm2 = glm(:(admit ~ gre + gpa + rank), dd, Bernoulli())
