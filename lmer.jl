const template = Formula(:(~ foo))      # for evaluating the lhs of r.e. terms

function lmer(f::Formula, fr::AbstractDataFrame; dofit=true)
    mf = ModelFrame(f, fr); df = mf.df; n = size(df,1)
    
    ## extract random-effects terms and check there is at least one
    re = filter(x->Meta.isexpr(x,:call) && x.args[1] == :|, mf.terms.terms)
    k = length(re); k > 0 || error("Formula $f has no random-effects terms")
    
    ## reorder terms by non-increasing number of levels
    gf = PooledDataVector[df[t.args[3]] for t in re]  # grouping factors
    p = sortperm(Int[length(f.pool) for f in gf]; rev=true)
    re = re[p]; gf = gf[p]

    ## create and fill vectors of matrices from the random-effects terms
    u = Array(Matrix{Float64},k); Xs = similar(u); lambda = similar(u)
    rowval = Array(Matrix{Int},k); inds = Array(Any,k); offset = 0; scalar = true
    for i in 1:k                    # iterate over random-effects terms
        t = re[i]
        if t.args[2] == 1 Xs[i] = ones(n,1); p = 1; lambda[i] = ones(1,1)
        else
            Xs[i] = (template.rhs=t.args[2]; ModelMatrix(ModelFrame(template, df))).m
            p = size(Xs[i],2); lambda[i] = eye(p)
        end
        if p > 1; scalar = false; end
        l = length(gf[i].pool); u[i] = zeros(p,l); nu = p*l; ii = gf[i].refs
        inds[i] = ii; rowval[i] = (reshape(1:nu,(p,l)) + offset)[:,ii]
        offset += nu
    end
    Ti = Int; if offset < typemax(Int32) Ti = Int32 end ## use 32-bit ints if possible

    X = ModelMatrix(mf); p = size(X.m,2); nz = hcat(Xs...)'; rv = vcat(rowval...)
    y = float(vector(model_response(mf)))
    local m
                                     # create the appropriate type of LMM object
    if k == 1 && scalar              # either LMMScalar1 or LMMVector1
        ## FIXME: Move this to an external constructor
        q = offset; Xt = X.m'; XtX = Xt*Xt'
        ZtZ = zeros(q); XtZ = zeros(p,q); Zty = zeros(q);
        Ztrv = convert(Vector{Ti},vec(rv)); Ztnz = vec(nz)
        for i in 1:n
            j = Ztrv[i]; z = Ztnz[i]
            ZtZ[j] += z*z; Zty[j] += z*y[i]; XtZ[:,j] += z*Xt[:,i]
        end
        m = LMMScalar1{Ti}(1., ones(q), cholfact(XtX), Xt, XtX,
                           XtZ, Xt*y, Ztrv, Ztnz, ZtZ, Zty,
                           zeros(p), zeros(n), zeros(q), y, false, false);
    else
        ## FIXME: another external constructor
        LambdatZt = CholmodSparse!(convert(Vector{Ti}, [1:size(nz,1):length(nz)+1]),
                                   convert(Vector{Ti}, vec(rv)),
                                   vec(nz), offset, n, 0)
        L = cholfact(LambdatZt,1.,true); pp = invperm(L.Perm + one(Ti))
        rowvalperm = Ti[pp[rv[i,j]] for i in 1:size(rv,1), j in 1:size(rv,2)]

        m = LMMGeneral(cholfact(LambdatZt,1.,true),LambdatZt,cholfact(eye(p)),
            X,Xs,X.m\y,inds,lambda,zeros(n),vec(rowvalperm),u,y,false,false)
    end
    dofit ? fit(m) : m
end
lmer(ex::Expr, fr::AbstractDataFrame) = lmer(Formula(ex), fr)

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
