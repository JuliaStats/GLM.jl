using GLM
using DataFrames
using Random
using CSV
using StatsBase
using RDatasets
Random.seed!(11)

y = rand(10)
x = rand(10,2)
wts = rand(10)
df = DataFrame(x, :auto)
df.y = y
df.wts = wts
lm1 = lm(x,y)
lmw = lm(x,y; wts = wts)
lmf = lm(@formula(y~x1+x2-1), df)
lmfw = lm(@formula(y~-1+x1+x2), df; wts = aweights(wts))
lmfw = lm(@formula(y~-1+x1+x2), df; wts = pweights(wts))
lmfw = lm(@formula(y~-1+x1+x2), df; wts = fweights(wts))

glm(@formula(y~-1+x1+x2), df, Normal, IdentityLink; wts = fweights(wts))

cooksdistance(lm1)



df = dataset("quantreg", "engel")
N = nrow(df)
df.weights = repeat(1:5, Int(N/5))
f = @formula(FoodExp ~ Income)
lm_model = lm(f, df, wts = FrequencyWeights(df.weights))
glm_model = glm(f, df, Normal(), wts = FrequencyWeights(df.weights))
@test isapprox(coef(lm_model), [154.35104595140706, 0.4836896390157505])
@test isapprox(coef(glm_model), [154.35104595140706, 0.4836896390157505])
@test isapprox(stderror(lm_model), [9.382302620120193, 0.00816741377772968])
@test isapprox(r2(lm_model), 0.8330258148644486)
@test isapprox(adjr2(lm_model), 0.832788298242634)
@test isapprox(vcov(lm_model), [88.02760245551447 -0.06772589439264813; 
                                -0.06772589439264813 6.670664781664879e-5])
@test isapprox(first(predict(lm_model)), 357.57694841780994)
@test isapprox(loglikelihood(lm_model), -4353.946729075838)
@test isapprox(loglikelihood(glm_model), -4353.946729075838)
@test isapprox(nullloglikelihood(lm_model), -4984.892139711452)
@test isapprox(mean(residuals(lm_model)), -5.412966629787718) 

lm_model = lm(f, df, wts = df.weights)
glm_model = glm(f, df, Normal(), wts = df.weights)
@test isapprox(coef(lm_model), [154.35104595140706, 0.4836896390157505])
@test isapprox(coef(glm_model), [154.35104595140706, 0.4836896390157505])
@test isapprox(stderror(lm_model), [9.382302620120193, 0.00816741377772968])
@test isapprox(r2(lm_model), 0.8330258148644486)
@test isapprox(adjr2(lm_model), 0.832788298242634)
@test isapprox(vcov(lm_model), [88.02760245551447 -0.06772589439264813; 
                                -0.06772589439264813 6.670664781664879e-5])
@test isapprox(first(predict(lm_model)), 357.57694841780994)
@test isapprox(loglikelihood(lm_model), -4353.946729075838)
@test isapprox(loglikelihood(glm_model), -4353.946729075838)
@test isapprox(nullloglikelihood(lm_model), -4984.892139711452)
@test isapprox(mean(residuals(lm_model)), -5.412966629787718) 



lm_model = lm(f, df, wts = aweights(df.weights))
glm_model = glm(f, df, Normal(), wts = aweights(df.weights))
@test isapprox(coef(lm_model), [154.35104595140706, 0.4836896390157505])
@test isapprox(coef(glm_model), [154.35104595140706, 0.4836896390157505])
@test isapprox(stderror(lm_model), [16.297055281313032, 0.014186793927918842])
@test isapprox(r2(lm_model), 0.8330258148644486)
@test isapprox(adjr2(lm_model), 0.8323091874604334)
@test isapprox(vcov(lm_model), [265.59401084217296   -0.20434035947652907; 
                                 -0.20434035947652907 0.00020126512195323495])
@test isapprox(first(predict(lm_model)), 357.57694841780994)
@test isapprox(loglikelihood(lm_model), -4353.946729075838)
@test isapprox(loglikelihood(glm_model), -4353.946729075838)
@test isapprox(nullloglikelihood(lm_model), -4984.892139711452)
@test isapprox(mean(residuals(lm_model)), -5.412966629787718) 