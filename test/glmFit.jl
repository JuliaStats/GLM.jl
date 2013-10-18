using Base.Test
using GLM

dobson = DataFrame(counts=[18.,17,15,20,10,20,25,13,12], outcome=gl(3,1,9), treatment=gl(3,3))
gm1 = glm(:(counts ~ outcome + treatment), dobson, Poisson());
@test_approx_eq deviance(gm1) 5.12914107700115

## Example from http://www.ats.ucla.edu/stat/r/dae/logit.htm
nm = download("http://www.ats.ucla.edu/stat/data/binary.csv", tempname())
df = within(readtable(nm),:(rank=compact(PooledDataArray(rank))))
rm(nm)

rteps = sqrt(eps())

gm2 = glm(:(admit ~ gre + gpa + rank), df, Binomial());
@test_approx_eq deviance(gm2) 458.5174924758994
@test sumsqdiff(coef(gm2), [-3.9899786606380734,0.0022644256521549043,0.8040374535155766,-0.6754428594116577,-1.3402038117481079,-1.5514636444657492]) < rteps

gm3 = glm(:(admit ~ gre + gpa + rank), df, Binomial(), ProbitLink());
@test_approx_eq deviance(gm3) 458.4131713833386
@test sumsqdiff(coef(gm3), [-2.3867922998680786,0.0013755394922972369,0.47772908362647015,-0.4154125854823675,-0.8121458010130356,-0.9359047862425298]) < rteps

gm4 = glm(:(admit ~ gre + gpa + rank), df, Binomial(), CauchitLink());
@test_approx_eq deviance(gm4) 459.3401112751141

gm5 = glm(:(admit ~ gre + gpa + rank), df, Binomial(), CloglogLink());
@test_approx_eq deviance(gm5) 458.89439629612616

