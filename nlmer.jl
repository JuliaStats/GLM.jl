# data for subject 1 in the theophylline experiment
# the time zero observation has been removed

const t = [0.25, 0.57, 1.12, 2.02, 3.82, 5.1, 7.03, 9.05, 12.12, 24.37]
const dose = 4.02
const conc = [2.84, 6.57, 10.5, 9.66, 8.58, 8.36, 7.47, 6.89, 5.94, 3.28]

function ff(x::Vector{Float64})         # parameters are lV, lka, and lCl
    expr1 = exp(x[1])
    expr2 = dose/expr1
    expr3 = exp(x[2])
    expr4 = exp(x[3])
    expr5 = expr4/expr1
    expr6 = expr3 - expr5
    expr7 = expr3/expr6
    expr8 = expr2 * expr7
    expr11 = exp(-expr5 * t)
    expr14 = exp(-expr3 * t)
    expr15 = expr11 - expr14
    expr18 = expr1^2
    expr19 = expr4 * expr1/expr18
    expr24 = expr6^2
    expctd = expr8 * expr15
    grad = Array(Float64,length(expctd), 3)
    grad[:,1] = expr8 .* (expr11 .* (expr19 .* t)) -
                         (expr2 .* (expr3 .* expr19/expr24) + dose .* expr1/expr18 .* expr7) .* expr15
    grad[:,2] = expr2 .* (expr7 - expr3 .* expr3/expr24) .* expr15 + expr8 .* (expr14 .* (expr3 .* t))
    grad[:,3] = expr2 .* (expr3 * expr5/expr24) .* expr15 - expr8 .* (expr11 .* (expr5 .* t))
    expctd, grad
end

const tol = 0.001
const minfac = 2. ^(-10)
const maxiter = 1000

function GN(obs::Vector{Float64}, f::Function, init::Vector{Float64})
    th = copy(init)
    p = length(th)
    converged = false
    n = length(obs)
    prev = Inf
    tolsq = square(tol)
    expd, grad = ff(th)
    resid = obs - expd
    oldrss = rss = (s=0.;for r in resid s += r*r end; s)
    for i in 1:maxiter
        qrf = qrfact(grad)
        qtr = copy(resid)
        Base.LinAlg.LAPACK.gemqrt!('L','T',qrf.vs,qrf.T,qtr)
        crit = (s=0.;for i in 1:p s+=square(qtr[i]) end;s)/oldss
        if crit < tolsq converged = true end
        inc = qrf\resid      # this is a repetition of Q'resid but ...
        f = 1.
        while f >= minfac
            expd, grad = ff(th + f*inc)
            resid = obs - expd
            rss = (s=0.;for r in resid s += r*r end; s)
            if rss < oldrss break end
            f *= 0.5                    # step-halving
        end
        if f < minfac error("step factor reduced below minimum, $minfac") end
        oldrss = rss
        th = th + f*inc
        if converged break end
    end
    th
end
