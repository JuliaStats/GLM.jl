## Types representing linear mixed models with vector-valued random effects

type VectorLMM1{Ti<:Integer} <: LinearMixedModel
    L::Array{Float64,3}
    LX::Matrix{Float64}
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
    mu::Vector{Float64}
    theta::Vector{Float64}
    u::Matrix{Float64}
    y::Vector{Float64}
    REML::Bool
    fit::Bool
end

function VectorLMM1{Ti<:Integer}(Xt::Matrix{Float64}, Ztrv::Vector{Ti},
                                 Ztnz::Matrix{Float64}, y::Vector{Float64})
    p,n = size(Xt)
    if length(Ztrv) != n || size(Ztnz,2) != n || length(y) != n
        error(string("Dimension mismatch: n = $n, ",
                     "lengths of Ztrv = $(length(Ztrv)), ",
                     "size(Ztnz,2) = $(size(Ztnz,2)), y = $(length(y))"))
    end
    k = size(Ztnz,1) # number of random effects per grouping factor level
    if k < 2 error("Use ScalarLMM1 instead of VectorLMM1") end
    urv = unique(Ztrv)
    if !isperm(urv) error("unique(Ztrv) must be a permutation") end
    nl = length(urv)
    ZtZ = zeros(k,k,nl); XtZ = zeros(p,k,nl); Zty  = zeros(k,nl)
    for j in 1:n
        i = Ztrv[j]; z = Ztnz[:,j]; 
        ZtZ[:,:,i] += z*z'; Zty[:,i] += y[j] * z; XtZ[:,:,i] += Xt[:,j]*z'
    end
    lower = vcat([(v=fill(-Inf,j);v[1]=0.;v) for j in k:-1:1]...)
    VectorLMM1{Ti}(zeros(k,k,nl), zeros(p,p), Xt, syrk('L','N',1.,Xt),
                   XtZ, Xt*y, Ztrv, Ztnz, ZtZ, Zty, zeros(p), lower,
                   zeros(n), float64(lower .== 0.), # initial value of theta
                   zeros(k,nl), y, false, false)
end        

size(m::VectorLMM1) = (length(m.y), length(m.beta), length(m.u), 1)
grplevels(m::VectorLMM1) = [((k,q)=size(m.L);div(q,k))]
Lmat(m::VectorLMM1) = [copy(m.L)]
LXmat(m::VectorLMM1) = tril(m.LX)
ranef!(m::VectorLMM1) = ranef(m, mkLambda(m))

function objective!(m::VectorLMM1, th::Vector{Float64})
    if any(th .< m.lower) error("theta = $th violates lower bounds") end
    n,p,q,t = size(m); k,nl = size(m.u)
    Lambda = mkLambda(k,th); ee = eye(k); ldL2 = 0.
    copy!(m.theta,th); copy!(m.L,m.ZtZ); copy!(m.LX,m.XtX)
    copy!(m.beta,m.Xty); copy!(m.u,m.Zty); 
    trmm!('L','L','T','N',1.,Lambda,reshape(m.L,(k,q)))
    trmm!('L','L','T','N',1.,Lambda,m.u) # Lambda'Z'y
    LXZ = Array(Float64,size(m.XtZ)); wL = Array(Float64,(k,k))
    wLXZ = Array(Float64,(p,k)); wu = zeros(k)
    for l in 1:nl
        wL[:] = m.L[:,:,l]                  # Lambda'(Z'Z)_l
        trmm!('R','L','N','N',1.,Lambda,wL) # Lambda'(Z'Z)_l*Lambda
        wL += ee
        _, info = potrf!('L',wL)       # l'th diagonal block of L_Z
        if bool(info)      # should not happen b/c +eye(k)
            error("Cholesky decomposition failed at l = $l")
        end
        m.L[:,:,l] = tril(wL)
        for j in 1:k ldL2 += 2.log(wL[j,j]) end
        wLXZ[:] = m.XtZ[:,:,l]
        trmm!('R','L','N','N',1.,Lambda,wLXZ) #(X'Z)_l*Lambda
        trsm!('R','L','T','N',1.,wL,wLXZ)# solve for LXZ_l
        LXZ[:,:,l] = wLXZ
        syrk!('L','N',-1.,wLXZ,1.,m.LX) # downdate X'X
        wu[:] = m.u[:,l]                # Lambda'(Z'y)_l
        trsv!('L','N','N',wL,wu)        # c_l = L_l\Lambda'(Z'y)_l
        m.u[:,l] = wu
        m.beta -= wLXZ*wu               # downdate X'y by LZX_l*c_l
    end
    _, info = potrf!('L',m.LX)          # form LX
    if bool(info) error("Downdated X'X is not positive definite") end
    potrs!('L',m.LX,m.beta)             # beta = (LX*LX')\(downdated X'y)
    gemv!('T',1.,m.Xt,m.beta,0.,m.mu)   # initialize mu to Xt'beta
    gemv!('T',-1.,reshape(LXZ,(p,q)),m.beta,1.,reshape(m.u,(q,))) # c - LXZ'beta
    for l in 1:nl trsv!('L','T','N', sub(m.L,1:k,1:k,l), sub(m.u,1:k,l)) end 
    bb = trmm('L','L','N','N',1.,Lambda,m.u) # b = Lambda * u
    for i in 1:n m.mu[i] += dot(bb[:,m.Ztrv[i]], m.Ztnz[:,i]) end
    fn = float64(n - (m.REML ? p : 0))
    obj = ldL2 + fn * (1. + log(2.pi * pwrss(m)/fn))
    if m.REML
        for i in 1:p obj += log(m.LX[i,i]) end
    end
    obj
end
sqrtwts!(m::VectorLMM1) = Float64[]
