abstract LinearMixedModel <: MixedModel
## Interface definition requires numeric vectors y and mu of the same length, n
## Numeric vector sqrtwts must have length 0 or n
## Float64 vectors theta and lower must have the same length

## Weighted residual sum of squares
function wrss(m::LinearMixedModel)
    s = zero(eltype(m.y))
    w = bool(length(m.sqrtwts))
    for i in 1:length(m.y)
        r = m.y[i] - m.mu[i]
        if w r /= sqrtwts[i] end
        s += r * r
    end
    s
end

## Penalized, weighted residual sum of squares
pwrss(m::LinearMixedModel) = wrss(m) + ussq(m)

## Optimization of objective using BOBYQA
function fit(m::LinearMixedModel, verbose::Bool)
    if !m.fit
        k = length(m.theta)
        opt = Opt(:LN_BOBYQA, k)
        ftol_abs!(opt, 1e-6)    # criterion on deviance changes
        xtol_abs!(opt, 1e-6)    # criterion on all parameter value changes
        lower_bounds!(opt, m.lower)
        function obj(x::Vector{Float64}, g::Vector{Float64})
            if length(g) > 0 error("gradient evaluations are not provided") end
            installtheta(m, x)
            updatemu(m)
            objective(m)
        end
        if verbose
            count = 0
            function vobj(x::Vector{Float64}, g::Vector{Float64})
                if length(g) > 0 error("gradient evaluations are not provided") end
                count += 1
                val = obj(x, g)
                println("f_$count: $val, $x")
                val
            end
            min_objective!(opt, vobj)
        else
            min_objective!(opt, obj)
        end
        fmin, xmin, ret = optimize(opt, m.theta)
        if verbose println(ret) end
        m.fit = true
    end
    m
end
fit(m::LinearMixedModel) = fit(m, false)      # non-verbose

function objective(m::LinearMixedModel)
    n, p, q = size(m)
    lnum = log(2pi * pwrss(m))
    if m.REML
        nmp = float(n - p)
        return logdetLRX(m) + nmp * (1 + lnum - log(nmp))
    end
    fn = float(n)
    logdetL(m) + fn * (1 + lnum - log(fn))
end

deviance(m::LinearMixedModel) = (m.REML=false; fit(m); objective(m, m.theta))

reml(m::LinearMixedModel) = (m.REML = true; m.fit = false; m)

function show(io::IO, m::LinearMixedModel)
    fit(m)
    REML = m.REML
    criterionstr = REML ? "REML" : "maximum likelihood"
    println(io, "Linear mixed model fit by $criterionstr")
    oo = objective(m)
    if REML
        println(io, " REML criterion: $oo")
    else
        println(io, " logLik: $(-oo/2), deviance: $oo")
    end
    vc = VarCorr(m)
    println("\n  Variance components: $vc")
    n, p, q = size(m)
    println("  Number of obs: $n; levels of grouping factors: $(grplevs(m))")
    println("  Fixed-effects parameters: $(fixef(m))")
end

abstract SimpleLinearMixedModel <: LinearMixedModel

function VarCorr(m::SimpleLinearMixedModel)
    fit(m)
    n, p, q = size(m)
    [m.theta .^ 2, 1.] * (pwrss(m)/float(n - (m.REML ? p : 0)))
end

function grplevs(m::SimpleLinearMixedModel)
    rv = m.Zt.rowval
    [length(unique(rv[i,:]))::Int for i in 1:size(rv,1)]
end

## Check validity and install a new value of theta;
##  update lambda, A
function installtheta(m::SimpleLinearMixedModel, theta::Vector{Float64})
    n, p, q = size(m)
    if length(theta) != length(m.theta) error("Dimension mismatch") end
    if any(theta .< m.lower)
        error("theta = $theta violates lower bounds $(m.lower)")
    end
    m.theta[:] = theta[:]               # copy in place
    for i in 1:length(theta)            # update Lambda (stored as a vector)
        fill!(m.lambda, theta[i], int(m.Zt.rowrange[i]))
    end
end
