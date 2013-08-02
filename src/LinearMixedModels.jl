## Base implementations of methods for the LinearMixedModel abstract type

## fit(m) -> m Optimization the objective using BOBYQA from the NLopt package
function fit(m::LinearMixedModel, verbose=false)
    if !isfit(m)
        th = theta(m); k = length(th)
        opt = Opt(:LN_BOBYQA, k)
        ftol_abs!(opt, 1e-6)    # criterion on deviance changes
        xtol_abs!(opt, 1e-6)    # criterion on all parameter value changes
        lower_bounds!(opt, lower(m))
        function obj(x::Vector{Float64}, g::Vector{Float64})
            length(g) == 0 || error("gradient evaluations are not provided")
            objective(solve!(theta!(m,x),true))
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
        fmin, xmin, ret = optimize(opt, th)
        if verbose println(ret) end
        m.fit = true
    end
    m
end

##  coef(m) -> current value of beta (can be a reference)
coef(m::LinearMixedModel) = m.beta

## coeftable(m) -> DataFrame : the coefficients table
## FIXME Create a type with its own show method for this type of table
function coeftable(m::LinearMixedModel)
    fe = fixef(m); se = stderr(m)
    DataFrame({fe, se, fe./se}, ["Estimate","Std.Error","z value"])
end

## deviance(m) -> Float64
deviance(m::LinearMixedModel) = m.fit && !m.REML ? objective(m) : NaN
        
##  fixef(m) -> current value of beta (can be a reference)
fixef(m::LinearMixedModel) = m.beta

##  isfit(m) -> Bool - Has the model been fit?
isfit(m::LinearMixedModel) = m.fit

## objective(m) -> deviance or REML criterion according to m.REML
function objective(m::LinearMixedModel)
     n,p,q,k = size(m); fn = float64(n - (m.REML ? p : 0))
    logdet(m,false) + fn*(1.+log(2.pi*pwrss(m)/fn)) + (m.REML ? logdet(m) : 0.)
end

## pwrss(m) -> penalized, weighted residual sum of squares
pwrss(m::LinearMixedModel) = rss(m) + sqrlenu(m)

rss(m::LinearMixedModel) = sqdiffsum(m.mu, m.y)

## scale(m) -> estimate, s, of the scale parameter
## scale(m,true) -> estimate, s^2, of the squared scale parameter
function scale(m::LinearMixedModel, sqr=false)
    n,p = size(m); ssqr = pwrss(m)/float64(n - (m.REML ? p : 0))
    sqr ? ssqr : sqrt(ssqr)
end

function show(io::IO, m::LinearMixedModel)
    fit(m); n, p, q, k = size(m); REML = m.REML
    println(io, string("Linear mixed model fit by ", REML ? "REML" : "maximum likelihood"))
    oo = objective(m)
    println(io, REML?" REML criterion: $oo":" logLik: $(round(-oo/2.,3)), deviance: $(round(oo,3))")
    println("\n  Variance components:")
    stddevs = vcat(std(m)...)
    println(io,"    Std. deviation scale:", [signif(x,5) for x in stddevs])
    println(io,"    Variance scale:", [signif(abs2(x),5) for x in stddevs])
    isscalar(m) || println(io,"    Correlations:\n", cor(m))
    println(io,"  Number of obs: $n; levels of grouping factors:", grplevels(m))
    println(io,"\n  Fixed-effects parameters:")
    tstrings = split(string(coeftable(m)),'\n')
    for i in 2:p+2 print(io,tstrings[i]); print(io,"\n") end
end

## stderr(m) -> standard errors of fixed-effects parameters
stderr(m::LinearMixedModel) = sqrt(diag(vcov(m)))

## vcov(m) -> estimated variance-covariance matrix of the fixed-effects parameters
vcov(m::LinearMixedModel) = scale(m,true) * inv(cholfact(m))

##  unsetfit!(m) -> m : unset the m.fit flag
unsetfit!(m::LinearMixedModel) = (m.fit = false; m)
