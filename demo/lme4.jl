## Some of the examples from the lme4 package for R

using RDatasets, GLM

ds = data("lme4", "Dyestuff")
fm1 = lmm(:(Yield ~ 1|Batch), ds);
println(fm1)

ds2 = data("lme4", "Dyestuff2")
fm1a = lmm(:(Yield ~ 1|Batch), ds2);
println(fm1a)

psts = data("lme4", "Pastes")
fm2 = lmm(:(strength ~ (1|sample) + (1|batch)), psts);
println(fm2)

pen = data("lme4", "Penicillin")
fm3 = lmm(:(diameter ~ (1|plate) + (1|sample)), pen);
println(fm3)

chem = data("mlmRev", "Chem97")
fm4 = lmm(:(score ~ (1|school) + (1|lea)), chem);
println(fm4)

fm5 = lmm(:(score ~ gcsecnt + (1|school) + (1|lea)), chem);
println(fm5)

@time fm6 = fit(reml!(lmm(:(score ~ gcsecnt + (1|school) + (1|lea)), chem)));
println(fm6)

inst = data("lme4", "InstEval")
@time fm7 = lmm(:(y ~ dept*service + (1|s) + (1|d)), inst);
println(fm7)

sleep = data("lme4", "sleepstudy")
@time fm8 = lmm(:(Reaction ~ Days + (Days | Subject)), sleep);
println(fm8)


