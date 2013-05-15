## Types representing linear mixed models with vector-valued random effects

type VectorLMM1{Ti<:Integer} <: LinearMixedModel
    L::Matrix{Float64}
    LX::Matrix{Float64}
    Xt::Matrix{Float64}
    XtX::Matrix{Float64}
    XtZ::Matrix{Float64}
    Xty::Vector{Float64}
    Ztrv::Vector{Ti}
    Ztnz::Matrix{Float64}
    ZtZ::Matrix{Float64}
    Zty::Vector{Float64}
    beta::Vector{Float64}
    lower::Vector{Float64}
    mu::Vector{Float64}
    theta::Vector{Float64}
    u::Vector{Float64}
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
    q = k * nl
    ZtZ = zeros(k,q); XtZ = zeros(p,q); Zty  = zeros(q)
    for j in 1:n
        i = Ztrv[j]; z = Ztnz[:,j]; ii = [1:k] + (i - 1)*k
        ZtZ[:,ii] += z*z'; Zty[ii] += y[j] * z; XtZ[:,ii] += Xt[:,j]*z'
    end
    lower = vcat([(v=fill(-Inf,j);v[1]=0.;v) for j in k:-1:1]...)
    VectorLMM1{Ti}(zeros(k,q), zeros(p,p), Xt, syrk('L','N',1.,Xt),
                   XtZ, Xt*y, Ztrv, Ztnz, ZtZ, Zty, zeros(p), lower,
                   zeros(n), float64(lower .== 0.), # initial value of theta
                   zeros(q), y, false, false)
end        
size(m::VectorLMM1) = (length(m.y), length(m.beta), length(m.u), 1)
grplevels(m::VectorLMM1) = [((k,q)=size(m.L);div(q,k))]
Lmat(m::VectorLMM1) = [copy(m.L)]
LXmat(m::VectorLMM1) = tril(m.LX)
function mkLambda(k::Integer,th::Vector{Float64})
    if length(th) != (k*(k+1))>>1
        error("length(th) = $(length(th)) should be $((k*(k+1))>>1) for k = $k")
    end
    tt = zeros(k*k); tt[bool(vec(tril(ones(k,k))))] = th
    reshape(tt, (k, k))
end
    
function objective!(m::VectorLMM1, th::Vector{Float64})
    if any(th .< m.lower) error("theta = $th violates lower bounds") end
    m.theta[:] = th; n,p,q,t = size(m); k,q = size(m.ZtZ); nl = div(q,k)
    Lambda = mkLambda(k,th); ee = eye(k)
    m.L[:] = m.ZtZ
    trmm!('L','L','T','N',1.,Lambda,m.L)
    m.LX[:] = m.XtX
    LXZ = Array(Float64, (p,q))
    m.beta[:] = m.Xty
    ldL2 = 0.; cols = 1:k; ee = eye(k)
    wu = Array(Float64,k); wL = Array(Float64,(k,k)) # work arrays
    wLXZ = Array(Float64,(p,k))
    for l in 1:nl
        wL[:] = m.L[:,cols]
        trmm!('R','L','N','N',1.,Lambda,wL)
        wL += ee
        _, info = potrf!('L',wL)
        if bool(info)
            error("Cholesky decomposition failed at l = $l")
        end
        m.L[:,cols] = tril(wL)
        for j in 1:k ldL2 += 2.log(wL[j,j]) end
        wLXZ[:] = m.XtZ[:,cols]
        trmm!('R','L','N','N',1.,Lambda,wLXZ)
        trsm!('R','L','T','N',1.,wL,wLXZ)
        syrk!('L','N',-1.,wLXZ,1.,m.LX)
        LXZ[:,cols] = wLXZ
        wu[:] = m.Zty[cols]
        trmv!('L','T','N',Lambda,wu)
        ccall(("dtrsv_",libblas), Void,
              (Ptr{Uint8}, Ptr{Uint8}, Ptr{Uint8}, Ptr{BlasInt},
               Ptr{Float64}, Ptr{BlasInt}, Ptr{Float64}, Ptr{BlasInt}),
              &'L', &'N', &'N', &k, wL, &k, wu, &1)
        m.u[cols] = wu
        m.beta -= wLXZ*wu
        cols += k
    end
    _, info = potrf!('L',m.LX)
    if bool(info) error("Downdated X'X is not positive definite") end
    potrs!('L',m.LX,m.beta)
    gemv!('T',-1.,LXZ,m.beta,1.,m.u)
    gemv!('T',1.,m.Xt,m.beta,0.,m.mu)
    cols = 1:k
    for l in 1:nl
        wu[:] = m.u[cols]
        wL[:] = m.L[:,cols]
        ccall(("dtrsv_",libblas), Void,
              (Ptr{Uint8}, Ptr{Uint8}, Ptr{Uint8}, Ptr{BlasInt},
               Ptr{Float64}, Ptr{BlasInt}, Ptr{Float64}, Ptr{BlasInt}),
              &'L', &'T', &'N', &k, wL, &k, wu, &1)
        m.u[cols] = wu
        cols += k
    end
    cols = [1:k]
    for i in 1:n m.mu[i] += dot(Lambda*m.u[cols + (m.Ztrv[i]-1)*k], m.Ztnz[:,i]) end
    fn = float64(n - (m.REML ? p : 0))
    obj = ldL2 + fn * (1. + log(2.pi * pwrss(m)/fn))
    if m.REML
        for i in 1:p obj += log(m.LX[i,i]) end
    end
    obj
end
sqrtwts!(m::VectorLMM1) = Float64[]
