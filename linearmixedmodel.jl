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
               X,Xs,X.m\y,inds,lambda,u,y,false,false)
end
lmer(ex::Expr, fr::AbstractDataFrame) = lmer(Formula(ex), fr)

##  size(m) -> n, p, q, t (lengths of y, beta, u and # of re terms)
size(m::LMMGeneral) = (length(m.y), length(m.beta), sum([length(u) for u in m.u]), length(m.u))

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

## solve!(m) -> m : solve PLS problem for u given current beta
## solve!(m,true) -> m : solve PLS problem for u and beta
function solve!(m::LMMGeneral,ubeta::Bool)
    u = Float64[]
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
    m
end
solve!(m::LMMGeneral) = solve!(m,false)

## linpred(m) -> linear predictor
## linpred(m,true) -> negative residuals (mu - y)
function linpred(m::LMMGeneral,minusy::Bool) # FIXME: incorporate an offset
    lp = gemv!('N',1.,m.X.m,m.beta,-1.,minusy?copy(m.y):zeros(length(m.y)))
    Xs = m.Xs; u = m.u; lm = m.lambda; inds = m.inds
    for i in 1:length(Xs)               # iterate over r.e. terms
        bb = trmm('L','L','N','N',1.,lm[i],u[i]);
        X = Xs[i]; ind = inds[i]
        for j in 1:length(lp), k in 1:size(X,2)
            lp[j] += bb[k,ind[j]] * X[j,k]
        end
    end
    lp
end
linpred(m::LMMGeneral) = linpred(m,false)

## sqrlenu(m) -> total squared length of m.u (the penalty in the PLS problem)
sqrlenu(m::LMMGeneral) = (s = 0.; for uu in m.u, u in uu s += u*u end; s)
    
## pwrss(m) -> penalized, weighted residual sum of squares
pwrss(m::LMMGeneral) = (s = sqrlenu(m); for r in linpred(m,true) s += r*r end; s)

## objective(m) -> deviance or REML criterion according to m.REML
function objective(m::LMMGeneral)
     n,p = size(m.X.m); fn = float64(n - (m.REML ? p : 0))
    logdet(m.L) + fn*(1.+log(2.pi*pwrss(m)/fn)) + (m.REML ? logdet(m.RX) : 0.)
end

## ltri(M) -> vector of elements from the lower triangle (column major order)    
function ltri(M::Matrix)
    m,n = size(M); m == n || error("size(M) = ($m,$n), should be square")
    if m == 1 return [M[1,1]] end
    r = Array(eltype(M), m*(m+1)>>1); pos = 1
    for i in 1:m, j in i:m r[pos] = M[i,j]; pos += 1 end; r
end
    
## theta(m) -> vector of variance-component parameters
theta(m::LMMGeneral) = vcat([ltri(M) for M in m.lambda]...)

## lower(m) -> lower bounds on elements of theta
lower(m::LMMGeneral) = [x==0.?-Inf:0. for x in vcat([ltri(eye(M)) for M in m.lambda]...)]

## Optimization of objective using BOBYQA
## According to convention the name should be fit! as this is a
## mutating function but I figure fit is obviously mutating and I
## wanted to piggy-back on the fit generic created in Distributions
function fit(m::LinearMixedModel, verbose::Bool)
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
fit(m::LinearMixedModel) = fit(m, false)      # non-verbose

##  RX(x) -> the Cholesky factor of the downdated X'X (can be a reference)
RX(m::LMMGeneral) = m.RX

##  fixef(m) -> current value of beta (can be a reference)
fixef(m::LMMGeneral) = m.beta

##  isfit(m) -> Bool - Has the model been fit?
isfit(m::LMMGeneral) = m.fit

##  reml(m) -> Bool : Is the REML criterion to be used?
reml(m::LMMGeneral) = m.REML
##  reml!(m) -> m : Set the REML criterion
##  reml!(m,false) -> m : Unset the REML criterion
reml!(m::LMMGeneral,v::Bool) = (m.REML = v; m)
reml!(m::LMMGeneral) = reml!(m,true)
##  unsetfit!(m) -> m : unset the m.fit flag
unsetfit!(m::LMMGeneral) = (m.fit = false; m)    
    
##  ranef(m) -> vector of matrices of random effects on the original scale
##  ranef(m,true) -> vector of matrices of random effects on the U scale
function ranef(m::LMMGeneral,uscale::Bool)
    uscale && return m.u
    Matrix{Float64}[m.lambda[i] * m.u[i] for i in 1:length(m.u)]
end
ranef(m::LMMGeneral) = ranef(m,false)

##  grplevels(m) -> vector of number of levels in random-effect terms
grplevels(m::LMMGeneral) = [size(u,2) for u in m.u]
    
## scale(m) -> estimate, s, of the scale parameter
## scale(m,true) -> estimate, s^2, of the squared scale parameter
function scale(m::LMMGeneral,sqr::Bool)
    n,p = size(m.X.m); ssqr = pwrss(m)/float64(n - (m.REML ? p : 0)); 
    sqr ? ssqr : sqrt(ssqr)
end
scale(m::LMMGeneral) = scale(m,false)

## std(m) -> Vector{Vector{Float64}} estimated standard deviations of variance components
std(m::LMMGeneral) = scale(m) .* [Float64[norm(l[i,:]) for i in 1:size(l,1)] for l in m.lambda]

## convert a lower Cholesky factor to a correlation matrix
function cc(c::Matrix{Float64})
    m,n = size(c); m == n || error("argument of size $(size(c)) should be square")
    m == 1 && return ones(1,1)
    std = broadcast(/, c, Float64[norm(c[i,:]) for i in 1:size(c,1)])
    std * std'
end

## cor(m) -> correlation matrices of variance components
cor(m::LMMGeneral) = [cc(l) for l in m.lambda]

## vcov(m) -> estimated variance-covariance matrix of the fixed-effects parameters
vcov(m::LMMGeneral) = scale(m,true) * inv(RX(m))

## isscalar(m) -> Bool : Are all the random-effects terms scalar?
isscalar(m::LMMGeneral) = all([size(l,1) == 1 for l in m.lambda])

## stderr(m) -> standard errors of fixed-effects parameters
stderr(m::LMMGeneral) = sqrt(diag(vcov(m)))

function show(io::IO, m::LinearMixedModel)
    fit(m)
    REML = m.REML
    println(io, string("Linear mixed model fit by ", REML ? "REML" : "maximum likelihood"))
    oo = objective(m)
    println(io, REML?" REML criterion: $oo":" logLik: $(-oo/2.), deviance: $oo")
    vc = VarCorr(m)
    println("\n  Variance components: $vc")
    n, p, q = size(m)
    println("  Number of obs: $n; levels of grouping factors: $(grplevels(m))")
    println("  Fixed-effects parameters: $(fixef(m))")
end

deviance(m::LMMGeneral) = objective(fit(reml(m) ? unsetfit!(reml!(m,false)) : m))
