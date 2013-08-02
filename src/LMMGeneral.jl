type LMMGeneral{Ti<:Union(Int32,Int64)} <: LinearMixedModel
    L::CholmodFactor{Float64,Ti}
    LambdatZt::CholmodSparse{Float64,Ti}
    RX::Cholesky{Float64}
    X::ModelMatrix{Float64}             # fixed-effects model matrix
    Xs::Vector{Matrix{Float64}}         # X_1,X_2,...,X_k
    beta::Vector{Float64}
    inds::Vector{Any}
    lambda::Vector{Matrix{Float64}}     # k lower triangular mats
    mu::Vector{Float64}
    rowvalperm::Vector{Ti}
    u::Vector{Matrix{Float64}}
    y::Vector{Float64}
    REML::Bool
    fit::Bool
end

function LMMGeneral{Ti<:Union(Int32,Int64)}(q::Integer, X::ModelMatrix, Xs::Vector{Matrix{Float64}},
                                            inds::Array, u::Vector{Matrix{Float64}}, rv::Matrix{Ti},
                                            y::Vector{Float64}, lambda::Vector{Matrix{Float64}})
    n,p = size(X.m); nz = hcat(Xs...)'
    LambdatZt = CholmodSparse!(convert(Vector{Ti}, [1:size(nz,1):length(nz)+1]),
                               vec(copy(rv)), vec(nz), q, n, 0)
    L = cholfact(LambdatZt,1.,true); pp = invperm(L.Perm + one(Ti))
    rowvalperm = Ti[pp[rv[i,j]] for i in 1:size(rv,1), j in 1:size(rv,2)]

    res = LMMGeneral{Ti}(L,LambdatZt,cholfact(eye(p)),X,Xs,zeros(p),inds,lambda,
        zeros(n),vec(rowvalperm),u,y,false,false)
    println(Ti); println(typeof(res))
    res
end

##  cholfact(x, RX=true) -> the Cholesky factor of the downdated X'X or LambdatZt
cholfact(m::LMMGeneral,RX=true) = RX ? m.RX : m.L

## cor(m) -> correlation matrices of variance components
cor(m::LMMGeneral) = [cc(l) for l in m.lambda]

## deviance!(m) -> Float64 : fit the model by maximum likelihood and return the deviance
deviance!(m::LMMGeneral) = objective(fit(reml!(m,false)))

##  grplevels(m) -> vector of number of levels in random-effect terms
grplevels(m::LMMGeneral) = [size(u,2) for u in m.u]

## isscalar(m) -> Bool : Are all the random-effects terms scalar?
isscalar(m::LMMGeneral) = all([size(l,1) == 1 for l in m.lambda])

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

## Logarithm of the determinant of the generator matrix for the Cholesky factor, L or RX
logdet(m::LMMGeneral,RX=true) = logdet(cholfact(m,RX))

## lower(m) -> lower bounds on elements of theta
lower(m::LMMGeneral) = [x==0.?-Inf:0. for x in vcat([ltri(eye(M)) for M in m.lambda]...)]

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

##  size(m) -> n, p, q, t (lengths of y, beta, u and # of re terms)
size(m::LMMGeneral) = (length(m.y), length(m.beta), sum([length(u) for u in m.u]), length(m.u))

## solve!(m) -> m : solve PLS problem for u given beta
## solve!(m,true) -> m : solve PLS problem for u and beta
function solve!(m::LMMGeneral, ubeta=false)
    local u                             # so u from both branches is accessible
    n,p,q,k = size(m)
    if ubeta
        nzmat = reshape(m.LambdatZt.nzval, (div(length(m.LambdatZt.nzval),n),n))
        scrm = similar(nzmat); RZX = Array(Float64, sum(length, m.u), p)
        rvperm = m.rowvalperm
        cu = solve(m.L, cmult!(nzmat, m.y, scrm, RZX[:,1], rvperm), CHOLMOD_L)
        ttt = solve(m.L,cmult!(nzmat, m.X.m, scrm, RZX, rvperm),CHOLMOD_L)
        potrf!('U',syrk!('U','T',-1.,ttt,1.,syrk!('U','T',1.,m.X.m,0.,m.RX.UL)))
        potrs!('U',m.RX.UL,gemv!('T',-1.,ttt,cu,1.,gemv!('T',1.,m.X.m,m.y,0.,m.beta)))
        gemv!('N',-1.,ttt,m.beta,1.,cu)
        u = solve(m.L,solve(m.L,cu,CHOLMOD_Lt),CHOLMOD_Pt)
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
