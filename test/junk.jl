rng = StableRNG(123)
df = DataFrame(x_1 = randn(rng, 10), x_2 = randn(rng, 10), y = randn(rng, 10), )
df.xx_1 = df.x_1
df.xx_2 = df.x_2
df.d = rand(rng, 0:1, 10)
df.w = rand(rng, 10)
frm0 = @formula(y ~ x_1 + x_2)
frm1 = @formula(y ~ x_1 + xx_2 + + x_2 + xx_1)
frmp0 = @formula(d ~ x_1 + x_2)
frmp1 = @formula(d ~ x_1 + xx_2 + + x_2 + xx_1)

probit0 = glm(frmp0, df, Binomial(), ProbitLink(), method=:qr, wts=df.w)

W = Diagonal(sqrt.(probit0.rr.wrkwt))
X = modelmatrix(probit0, weighted=false)

lm0 = lm(frm0, df, wts=df.w)
X = modelmatrix(probit0, weighted=false)
W = Diagonal(sqrt.(probit0.rr.wts))
v1 = diag(W*X*inv(X'W*W*X)*X'W)
v2 = diag(X*inv(X'W*W*X)*X').*probit0.rr.wts

pp = lm0.pp
rnk = rank(pp.chol)

X/pp.chol

p = pp.chol.p[1:rnk]
sum(x->x^2, view(X, :, p)/view(pp.chol.U, p, p), dims=2)

