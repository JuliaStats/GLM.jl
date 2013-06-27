abstract LinearMixedModel <: MixedModel

type LMMGeneral{Ti<:Union(Int32,Int64)} <: LinearMixedModel
    L::CholmodFactor{Float64,Ti}
    LambdatZt::CholmodSparse{Float64,Ti}
    RX::Cholesky{Float64}
    X::ModelMatrix                      # fixed-effects model matrix
    Xs::Vector{Matrix{Float64}}         # X_1,X_2,...,X_k
    beta::Vector{Float64}
    inds::Vector{Any}
    lambda::Vector{Matrix{Float64}}     # k lower triangular mats
    mu::Vector{Float64}
    u::Vector{Matrix{Float64}}
    y::Vector{Float64}
    REML::Bool
    fit::Bool
end

const template = Formula(:(~ foo))      # for evaluating the lhs of r.e. terms

function lmer(f::Formula, fr::AbstractDataFrame)
    mf = ModelFrame(f, fr); df = mf.df; n = size(df,1)

    ## extract random-effects terms and check there is at least one
    re = filter(x->Meta.isexpr(x,:call) && x.args[1] == :|, mf.terms.terms)
    k = length(re); k > 0 || error("Formula $f has no random-effects terms")

    ## reorder terms by non-increasing number of levels
    gf = PooledDataVector[df[t.args[3]] for t in re]  # grouping factors
    p = sortperm(Int[length(f.pool) for f in gf],Sort.Reverse); re = re[p]; gf = gf[p]

    ## create and fill vectors of matrices from the random-effects terms
    u = Array(Matrix{Float64},k); Xs = similar(u); lambda = similar(u)
    rowval = Array(Matrix{Int},k); inds = Array(Any,k); offset = 0
    for i in 1:k                    # iterate over random-effects terms
        t = re[i]
        if t.args[2] == 1 Xs[i] = ones(n,1); p = 1; lambda[i] = ones(1,1)
        else
            Xs[i] = (template.rhs=t.args[2]; ModelMatrix(ModelFrame(template, df))).m
            p = size(Xs[i],2); lambda[i] = eye(p)
        end
        l = length(gf[i].pool); u[i] = zeros(p,l); nu = p*l; ii = gf[i].refs
        inds[i] = ii; rowval[i] = (reshape(1:nu,(p,l)) + offset)[:,ii]
        offset += nu
    end
    Ti = Int; if offset < typemax(Int32) Ti = Int32 end ## use 32-bit ints if possible

    ## create the LMMGeneral object
    X = ModelMatrix(mf); p = size(X.m,2); nz = hcat(Xs...)'
    LambdatZt = CholmodSparse!(convert(Vector{Ti}, [1:size(nz,1):length(nz)+1]),
                               convert(Vector{Ti}, vec(vcat(rowval...))),
                               vec(nz), offset, n, 0)
    y = float(vector(model_response(mf)))
    LMMGeneral(cholfact(LambdatZt,1.,true),LambdatZt,cholfact(eye(p)),
               X,Xs,X.m\y,inds,lambda,zeros(n),u,y,false,false)
end
lmer(ex::Expr, fr::AbstractDataFrame) = lmer(Formula(ex), fr)

##  cholfact(x, RX=true) -> the Cholesky factor of the downdated X'X or LambdatZt
cholfact(m::LMMGeneral,RX=true) = RX ? m.RX : m.L

##  coef(m) -> current value of beta (can be a reference)
coef(m::LMMGeneral) = m.beta

## coeftable(m) -> DataFrame : the coefficients table
## FIXME Create a type with its own show method for this type of table
function coeftable(m::LMMGeneral)
    fe = fixef(m); se = stderr(m)
    DataFrame({fe, se, fe./se}, ["Estimate","Std.Error","z value"])
end

## convert a lower Cholesky factor to a correlation matrix
function cc(c::Matrix{Float64})
    m,n = size(c); m == n || error("argument of size $(size(c)) should be square")
    m == 1 && return ones(1,1)
    std = broadcast(/, c, Float64[norm(c[i,:]) for i in 1:size(c,1)])
    std * std'
end

## cor(m) -> correlation matrices of variance components
cor(m::LMMGeneral) = [cc(l) for l in m.lambda]

## deviance!(m) -> Float64 : fit the model by maximum likelihood and return the deviance
deviance!(m::LMMGeneral) = objective(fit(reml!(m,false)))

## deviance(m) -> Float64
deviance(m::LMMGeneral) = isfit(m) && !isreml(m) ? objective(m) : NaN

## fit(m) -> m Optimization the objective using BOBYQA from the NLopt package
function fit(m::LinearMixedModel, verbose=false)
    if !isfit(m)
        k = length(theta(m))
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
        fmin, xmin, ret = optimize(opt, theta(m))
        if verbose println(ret) end
        m.fit = true
    end
    m
end

##  fixef(m) -> current value of beta (can be a reference)
fixef(m::LMMGeneral) = m.beta

##  grplevels(m) -> vector of number of levels in random-effect terms
grplevels(m::LMMGeneral) = [size(u,2) for u in m.u]

##  isfit(m) -> Bool - Has the model been fit?
isfit(m::LMMGeneral) = m.fit
    
## isscalar(m) -> Bool : Are all the random-effects terms scalar?
isscalar(m::LMMGeneral) = all([size(l,1) == 1 for l in m.lambda])

##  isreml(m) -> Bool : Is the REML criterion to be used?
isreml(m::LMMGeneral) = m.REML

## linpred!(m) -> update mu
function linpred!(m::LMMGeneral)
    gemv!('N',1.,m.X.m,m.beta,0.,m.mu)  # initialize mu to X*beta
    Xs = m.Xs; u = m.u; lm = m.lambda; inds = m.inds; mu = m.mu
    for i in 1:length(Xs)               # iterate over r.e. terms
        X = Xs[i]; ind = inds[i]
        if size(X,2) == 1 fma!(mu, (lm[i][1,1]*u[i])[:,ind], X[:,1])
        else
            add!(mu,sum(trmm('L','L','N','N',1.0,lm[i],u[i])[:,ind]' .* X, 2))
        end
    end
    m
end

## lower(m) -> lower bounds on elements of theta
lower(m::LMMGeneral) = [x==0.?-Inf:0. for x in vcat([ltri(eye(M)) for M in m.lambda]...)]

## objective(m) -> deviance or REML criterion according to m.REML
function objective(m::LMMGeneral)
     n,p = size(m.X.m); fn = float64(n - (m.REML ? p : 0))
    logdet(m.L) + fn*(1.+log(2.pi*pwrss(m)/fn)) + (m.REML ? logdet(m.RX) : 0.)
end

## pwrss(m) -> penalized, weighted residual sum of squares
pwrss(m::LMMGeneral) = sqrlenu(m) + sqdiffsum(m.mu, m.y)

##  ranef(m) -> vector of matrices of random effects on the original scale
##  ranef(m,true) -> vector of matrices of random effects on the U scale
function ranef(m::LMMGeneral, uscale=false)
    uscale && return m.u
    Matrix{Float64}[m.lambda[i] * m.u[i] for i in 1:length(m.u)]
end

##  reml!(m,v=true) -> m : Set m.REML to v.  If m.REML is modified, unset m.fit
function reml!(m::LMMGeneral,v=true)
    v == m.REML && return m
    m.REML = v; m.fit = false
    m
end
    
## scale(m) -> estimate, s, of the scale parameter
## scale(m,true) -> estimate, s^2, of the squared scale parameter
function scale(m::LMMGeneral, sqr=false)
    n,p = size(m.X.m); ssqr = pwrss(m)/float64(n - (m.REML ? p : 0)); 
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

##  size(m) -> n, p, q, t (lengths of y, beta, u and # of re terms)
size(m::LMMGeneral) = (length(m.y), length(m.beta), sum([length(u) for u in m.u]), length(m.u))

## solve!(m) -> m : solve PLS problem for u given current beta
## solve!(m,true) -> m : solve PLS problem for u and beta
function solve!(m::LMMGeneral, ubeta=false)
    local u                             # so u from both branches is accessible
    if ubeta
        ltzty = m.LambdatZt * m.y
        cu = solve(m.L,solve(m.L,ltzty,CHOLMOD_P), CHOLMOD_L)
        RZX = solve(m.L,solve(m.L,m.LambdatZt * m.X.m,CHOLMOD_P),CHOLMOD_L).mat
        potrf!('U',syrk!('U','T',-1.,RZX,1.,syrk!('U','T',1.,m.X.m,0.,m.RX.UL)))
        potrs!('U',m.RX.UL,gemv!('T',-1.,RZX,vec(cu.mat),1.,gemv!('T',1.,m.X.m,m.y,0.,m.beta)))
        gemv!('N',-1.,RZX,m.beta,1.,vec(cu.mat))
        u = vec(solve(m.L,solve(m.L,cu,CHOLMOD_Lt),CHOLMOD_Pt).mat)
    else
        u = vec(solve(m.L,m.LambdatZt * gemv!('N',-1.0,m.X.m,m.beta,1.0,copy(m.y))).mat)
    end
    pos = 0
    for i in 1:length(m.u)
        ll = length(m.u[i])
        m.u[i] = reshape(sub(u,pos+(1:ll)), size(m.u[i]))
        pos += ll
    end
    linpred!(m)
end

## sqrlenu(m) -> total squared length of m.u (the penalty in the PLS problem)
sqrlenu(m::LMMGeneral) = sum([mapreduce(Abs2(),Add(),u) for u in m.u])

## std(m) -> Vector{Vector{Float64}} estimated standard deviations of variance components
std(m::LMMGeneral) = scale(m)*push!(Vector{Float64}[vec(vnorm(l,2,1)) for l in m.lambda],[1.])

## stderr(m) -> standard errors of fixed-effects parameters
stderr(m::LMMGeneral) = sqrt(diag(vcov(m)))

## ltri(M) -> vector of elements from the lower triangle (column major order)    
function ltri(M::Matrix)
    m,n = size(M); m == n || error("size(M) = ($m,$n), should be square")
    if m == 1 return [M[1,1]] end
    r = Array(eltype(M), m*(m+1)>>1); pos = 1
    for i in 1:m, j in i:m r[pos] = M[i,j]; pos += 1 end; r
end
    
## theta(m) -> vector of variance-component parameters
theta(m::LMMGeneral) = vcat([ltri(M) for M in m.lambda]...)

##  theta!(m,th) -> m : install new value of theta, update LambdatZt and L 
function theta!(m::LMMGeneral, th::Vector{Float64})
    n = length(m.y)
    nzmat = reshape(m.LambdatZt.nzval, (div(length(m.LambdatZt.nzval),n),n))
    lambda = m.lambda; Xs = m.Xs; tpos = 1; roff = 0 # position in th, row offset
    for kk in 1:length(Xs)
        T = lambda[kk]; p = size(T,1) # size of i'th template matrix
        for j in 1:p, i in j:p        # fill lower triangle from th
            T[i,j] = th[tpos]; tpos += 1
            i == j && T[i,j] < 0. && error("Negative diagonal element in T")
        end
        gemm!('T','T',1.,T,Xs[kk],0.,sub(nzmat,roff+(1:p),1:n))
        roff += p
    end
    cholfact!(m.L,m.LambdatZt,1.)
    m
end

##  unsetfit!(m) -> m : unset the m.fit flag
unsetfit!(m::LMMGeneral) = (m.fit = false; m)    

## vcov(m) -> estimated variance-covariance matrix of the fixed-effects parameters
vcov(m::LMMGeneral) = scale(m,true) * inv(cholfact(m))
