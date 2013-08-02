## A linear mixed model with a single vector-valued random-effects term

type LMMVector1{Ti<:Integer} <: LinearMixedModel
    ldL2::Float64
    L::Array{Float64,3}
    RX::Cholesky{Float64}
    Xt::Matrix{Float64}
    XtX::Matrix{Float64}
    XtZ::Array{Float64,3}
    Xty::Vector{Float64}
    Ztrv::Vector{Ti}
    Ztnz::Matrix{Float64}
    ZtZ::Array{Float64,3}
    Zty::Matrix{Float64}
    beta::Vector{Float64}
    lower::Vector{Float64}
    lambda::Matrix{Float64}
    mu::Vector{Float64}
    u::Matrix{Float64}
    y::Vector{Float64}
    REML::Bool
    fit::Bool
end

function LMMVector1{Ti<:Integer}(Xt::Matrix{Float64}, Ztrv::Vector{Ti},
                                 Ztnz::Matrix{Float64}, y::Vector{Float64})
    p,n = size(Xt)
    n == length(Ztrv) == size(Ztnz,2) == length(y) ||
        error(string("Dimension mismatch: n = $n, ",
                     "lengths of Ztrv = $(length(Ztrv)), ",
                     "size(Ztnz,2) = $(size(Ztnz,2)), y = $(length(y))"))
    k = size(Ztnz,1) # number of random effects per grouping factor level
    k > 1 || error("Use ScalarLMM1 instead of VectorLMM1")
    urv = unique(Ztrv)
    isperm(urv) || error("unique(Ztrv) must be a permutation")
    nl = length(urv)
    ZtZ = zeros(k,k,nl); XtZ = zeros(p,k,nl); Zty  = zeros(k,nl)
    for j in 1:n
        i = Ztrv[j]; z = Ztnz[:,j]
        ZtZ[:,:,i] += z*z'; Zty[:,i] += y[j] * z; XtZ[:,:,i] += Xt[:,j]*z'
    end
    lower = [x == 0. ? -Inf : 0. for x in ltri(eye(k))]
    XtX = syrk('U','N',1.,Xt)
    LMMVector1(0., zeros(k,k,nl), cholfact(XtX,:U), Xt, XtX,
               XtZ, Xt*y, Ztrv, Ztnz, ZtZ, Zty, zeros(p), lower, eye(k),
               zeros(n), zeros(k,nl), y, false, false)
end

cholfact(m::LMMVector1,RX=TRUE) = RX ? m.RX : error("not yet written")

grplevels(m::LMMVector1) = [m.size(m.u,2)]

isscalar(m::LMMVector1) = false

## linpred!(m) -> m   -- update mu
function linpred!(m::LMMVector1)
    gemv!('T',1.,m.Xt,m.beta,0.,m.mu)   # initialize to X*beta
    bb = trmm('L','L','N','N',1.,m.lambda,m.u) # b = Lambda * u
    for i in 1:length(m.mu) m.mu[i] += dot(bb[:,m.Ztrv[i]], m.Ztnz[:,i]) end
    m
end

## Logarithm of the determinant of the generator matrix for the Cholesky factor, RX or L
logdet(m::LMMVector1,RX=true) = RX ? logdet(m.RX) : m.ldL2
    
lower(m::LMMVector1) = m.lower

##  ranef(m) -> vector of matrices of random effects on the original scale
##  ranef(m,true) -> vector of matrices of random effects on the U scale
function ranef(m::LMMVector1, uscale=false)
    uscale && return [m.u]
    [trmm('L','L','N','N',1.,m.lambda,m.u)]
end
    
size(m::LMMVector1) = (length(m.y), length(m.beta), length(m.u), 1)

## solve!(m) -> m : solve PLS problem for u given beta
## solve!(m,true) -> m : solve PLS problem for u and beta
function solve!(m::LMMVector1, ubeta=false)
    n,p,q = size(m); k,nl = size(m.u); copy!(m.u,m.Zty)
    trmm!('L','L','T','N',1.,m.lambda,m.u)
    for l in 1:nl                       # cu := L^{-1} Lambda'Z'y
        trsv!('L','N','N',m.L[:,:,l], m.u[:,l])
    end
    if ubeta
        copy!(m.beta,m.Xty); copy!(m.RX.UL, m.XtX)
        LXZ = Array(Float64,size(m.XtZ)); wL = similar(m.lambda)
        wLXZ = Array(Float64,(p,k)); wu = zeros(k)
        for l in 1:nl
            copy!(wL, m.L[:,:,l]); copy!(wLXZ, m.XtZ[:,:,l])
            trmm!('R','L','N','N',1.,m.lambda,wLXZ) #(X'Z)_l*lambda
            trsm!('R','L','T','N',1.,wL,wLXZ)       # solve for LXZ_l
            copy!(LXZ[:,:,l],wLXZ)
            syrk!('U','N',-1.,wLXZ,1.,m.RX.UL) # downdate X'X
            m.beta -= wLXZ*m.u[:,l]     # downdate X'y by LZX_l*c_l
        end
        _, info = potrf!('U',m.RX.UL)   # update RX
        bool(info) && error("Downdated X'X is not positive definite")
        solve!(m.RX,m.beta)             # beta = (LX*LX')\(downdated X'y)
        for l in 1:nl                   # downdate cu
            gemv!('N',-1.,LXZ[:,:,l],m.beta,1.,m.u[:,l])
        end
    end
    for l in 1:nl
        trsv!('L','T','N',m.L[:,:,l], m.u[:,l])
    end
    linpred!(m)
end        

## sqrlenu(m) -> total squared length of m.u (the penalty in the PLS problem)
sqrlenu(m::LMMVector1) = sqsum(m.u)

theta(m::LMMVector1) = ltri(m.lambda)

##  theta!(m,th) -> m : install new value of theta, update L 
function theta!(m::LMMVector1, th::Vector{Float64})
    all(th .>= m.lower) || error("theta = $th violates lower bounds")
    n,p,q,t = size(m); k,nl = size(m.u); pos = 1
    for j in 1:k, i in j:k
        m.lambda[i,j] = th[pos]; pos += 1
    end
    ldL2 = 0.; copy!(m.L,m.ZtZ); wL = similar(m.lambda)
    trmm!('L','L','T','N',1.,m.lambda,reshape(m.L,(k,q)))
    for i in 1:nl
        copy!(wL, m.L[:,:,i])                 # lambda'(Z'Z)_l
        trmm!('R','L','N','N',1.,m.lambda,wL) # lambda'(Z'Z)_l*lambda
        for j in 1:k; wL[j,j] += 1.; end      # Inflate the diagonal
        _, info = potrf!('L',wL)        # i'th diagonal block of L_Z
        bool(info) && error("Cholesky decomposition failed at i = $i")
        m.L[:,:,i] = tril(wL)
        for j in 1:k; ldL2 += 2.log(wL[j,j]); end
    end
    m.ldL2 = ldL2
    m
end
