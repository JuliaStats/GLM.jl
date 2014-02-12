using Base.Test
using GLM

## Formaldehyde data from the R Datasets package
form = DataFrame(Carb=[0.1,0.3,0.5,0.6,0.7,0.9],OptDen=[0.086,0.269,0.446,0.538,0.626,0.782])
lm1 = lm(OptDen ~ Carb, form)
@test_approx_eq coef(lm1) linreg(convert(Array,form[:Carb]),convert(Array,form[:OptDen]))

dobson = DataFrame(Counts=[18.,17,15,20,10,20,25,13,12], Outcome=gl(3,1,9), Treatment=gl(3,3))
gm1 = glm(Counts ~ Outcome + Treatment, dobson, Poisson());
@test_approx_eq deviance(gm1) 5.12914107700115
@test_approx_eq coef(gm1)[1:3] [3.044522437723423,-0.45425527227759555,-0.29298712468147375]

## Example from http://www.ats.ucla.edu/stat/r/dae/logit.htm
nm = download("http://www.ats.ucla.edu/stat/data/binary.csv", tempname())
df = within(readtable(nm),:(rank=compact(PooledDataArray(rank))))
rm(nm)

rteps = sqrt(eps())

gm2 = glm(admit ~ gre + gpa + rank, df, Binomial())
@test_approx_eq deviance(gm2) 458.5174924758994
@test sumsqdiff(coef(gm2), [-3.9899786606380734,0.0022644256521549043,0.8040374535155766,-0.6754428594116577,-1.3402038117481079,-1.5514636444657492]) < rteps

gm3 = glm(admit ~ gre + gpa + rank, df, Binomial(), ProbitLink())
@test_approx_eq deviance(gm3) 458.4131713833386
@test sumsqdiff(coef(gm3), [-2.3867922998680786,0.0013755394922972369,0.47772908362647015,-0.4154125854823675,-0.8121458010130356,-0.9359047862425298]) < rteps

gm4 = glm(admit ~ gre + gpa + rank, df, Binomial(), CauchitLink())
@test_approx_eq deviance(gm4) 459.3401112751141

gm5 = glm(admit ~ gre + gpa + rank, df, Binomial(), CloglogLink())
@test_approx_eq deviance(gm5) 458.89439629612616

