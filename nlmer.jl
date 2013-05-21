# data for subject 1 in the theophylline experiment
# the time zero observation has been removed
using Base.LinAlg.BLAS: trsv!
using Base.LinAlg.LAPACK: gemqrt!,geqrt3!, potri!
import Distributions.fit, Base.show
using DataFrames

const conc = [2.84, 6.57, 10.5, 9.66, 8.58, 8.36, 7.47, 6.89, 5.94, 3.28]

## Create an abstract class NonlinearRegModel, perhaps with abstract subclasses and with
## concrete classes representing specific models.

## Overwrite eta with expected response at x
## Optionally, overwrite jac with the Jacobian matrix
## The jac code is from symbolic differentiation with common subexpression elimination
function ff{T<:Float64}(x::Vector{T}, eta::Vector{T}, jac::Matrix{T}) # parameters are lV, lka, and lCl
    const t = [0.25, 0.57, 1.12, 2.02, 3.82, 5.1, 7.03, 9.05, 12.12, 24.37]
    const dose = 4.02
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
    eta[:] = expr8 * expr15
    if size(jac,2) == length(x)
        expr18 = expr1^2
        expr19 = expr4 * expr1/expr18
        expr24 = expr6^2
        jac[:,1] = expr8 .*(expr11 .*(expr19 .*t)) -
                            (expr2 .*(expr3 .*expr19/expr24) + dose .*expr1/expr18 .*expr7).*expr15
        jac[:,2] = expr2 .*(expr7 - expr3 .*expr3/expr24) .*expr15 + expr8 .*(expr14 .*(expr3 .*t))
        jac[:,3] = expr2 .*(expr3*expr5/expr24) .*expr15 - expr8 .*(expr11 .*(expr5 .*t))
    end
end

type NonlinearLS                    # nonlinear least squares problems
    pars::Vector{Float64}
    incr::Vector{Float64}
    obs::Vector{Float64}
    expctd::Vector{Float64}             # expected response
    resid::Vector{Float64}
    qtr::Vector{Float64}
    jacob::Matrix{Float64}              # Jacobian matrix
    qr::QR{Float64}
    f::Function             # overwrites expctd and, optionally, jacob
    rss::Float64            # residual sum of squares at last successful iteration
    tolsqr::Float64         # squared tolerance for orthogonality criterion
    minfac::Float64
    mxiter::Int
    fit::Bool
end

function NonlinearLS(f::Function, obs::Vector{Float64}, init::Vector{Float64})
    n = length(obs); p = length(init);
    expctd = copy(obs); jacob = Array(Float64, n, p);
    f(init, expctd, jacob) # check that the evaluation of f works
    resid = obs - expctd
    NonlinearLS(init, zero(init), obs, expctd, resid, zero(resid), jacob,
                qrfact(jacob), f, sum(resid.^2), 1e-8, 0.5^10, 1000, false)
end

# evaluate expected response and residual at m.pars + fac * m.incr
# return residual sum of squares
function updtres(m::NonlinearLS, fac::Float64)
    m.f(m.pars + fac * m.incr, m.expctd, Array(Float64,(0,0)))
    s = 0.; r = m.resid; o = m.obs; e = m.expctd
    for i in 1:length(r)
        ri = o[i] - e[i]
        s += ri * ri
        r[i] = ri
    end
    s
end
    
# Create the QR factorization, qtr = Q'resid, solve for the increment and
# return the numerator of the squared convergence criterion
function qtr(m::NonlinearLS)
    vs = m.qr.vs; inc = m.incr; qt = m.qtr
    copy!(vs, m.jacob); copy!(qt, m.resid)
    _, T = geqrt3!(vs)
    copy!(m.qr.T,T)
    gemqrt!('L','T',vs,T,qt)
    s = 0.; p = size(vs,2)
    for i in 1:p qti = qt[i]; s += qti * qti; inc[i] = qti  end
    trsv!('U','N','N',sub(vs,1:p,1:p),inc)
    s
end

function gnfit(m::NonlinearLS)          # Gauss-Newton nonlinear least squares
    if !m.fit
        converged = false; rss = m.rss
        for i in 1:m.mxiter
            crit = qtr(m)/m.rss # evaluate increment and orthogonality cvg. crit.
            converged = crit < m.tolsqr
            f = 1.
            while f >= m.minfac
                rss = updtres(m,f)
                if rss < m.rss break end
                f *= 0.5                    # step-halving
            end
            if f < m.minfac
                error("Failure to reduce rss at $(m.pars) with incr = $(m.incr) and minfac = $(m.minfact)")
            end
            m.rss = rss
            m.pars += f * m.incr
            if converged break end
            m.f(m.pars, m.expctd, m.jacob)  # evaluate Jacobian
        end
        if !converged error("failure to converge in $(m.mxiter) iterations") end
        m.fit = true
    end
    m
end

function show(io::IO, m::NonlinearLS)
    gnfit(m)
    n,p = size(m.jacob)
    s2 = m.rss/float(n-p)
    varcov = s2 * symmetrize!(potri!('U', m.qr.vs[1:p,:])[1],'U')
    stderr = sqrt(diag(varcov))
    t_vals = m.pars./stderr
    println(io, "Model fit by nonlinear least squares to $n observations\n")
    println(io, DataFrame(parameter=m.pars,stderr=stderr,t_value=m.pars./stderr))
    println("Residual sum of squares at estimates = $(m.rss)")
    println("Residual standard error = $(sqrt(s2)) on $(n-p) degrees of freedom")
end
