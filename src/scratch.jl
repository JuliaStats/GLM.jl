using GLM
using DataFrames

y = rand(10)
x = rand(10,2)
wts = rand(10)
df = DataFrame(x, :auto)
df.y = y
df.wts = wts
lm1 = lm(x,y)
lmw = lm(x,y; wts = wts)
lmf = lm(@formula(y~x1+x2), df)
lmfw = lm(@formula(y~x1+x2), df; wts = wts)
glm(x, y)


cooksdistance(lm)



