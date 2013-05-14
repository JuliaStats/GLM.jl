## Types representing linear mixed models with vector-valued random effects

#using RDatasets, NLopt
#using Base.LinAlg.BLAS: gemv!, syrk!, syrk, trmv!, trmm!, trmm, trsm!
#using Base.LinAlg.LAPACK: potrf!, potrs!
#import Base.size

type VectorLMM1{Ti<:Integer} #<: LinearMixedModel
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
        ZtZ[:,ii] += z*z'; Zty[ii] += y[j] * z; XtZ[:,ii] += z*Xt[:,j]'
    end
    lower = vcat([(v=fill(-Inf,j);v[1]=0.;v) for j in k:-1:1]...)
    VectorLMM1{Ti}(zeros(k,q), zeros(k,k), Xt, syrk('L','N',1.,Xt),
                   XtZ, Xt*y, Ztrv, Ztnz, ZtZ, Zty, zeros(p), lower,
                   zeros(n), float64(lower .== 0.), # initial value of theta
                   zeros(q), y, false, false)
end        
size(m::VectorLMM1) = (length(m.y), length(m.beta), length(m.u), 1)
grplevels(m::VectorLMM1) = [size(m.L,3)]
Lmat(m::VectorLMM1) = [copy(m.L)]
LXmat(m::VectorLMM1) = copy(m.LX)
function objective!(m::VectorLMM1, th::Vector{Float64})
    if any(th .< m.lower) error("theta = $th violates lower bounds") end
    m.theta[:] = th; n,p,q,t = size(m); L = m.L; k,q = size(L); nl = div(q,k)
    tmp = zeros(k*k); tmp[bool(vec(tril(ones(k,k))))] = th
    Lambda = reshape(tmp, (k,k)); ee = eye(k)
    L[:] = m.ZtZ
    m.u[:] = m.Zty
    trmm!('L','L','T','N',1.,Lambda,L)
    LXZ = copy(m.XtZ)
    m.LX[:] = m.XtX
    ldL2 = 0.
    cols = 1:k
    m.beta[:] = m.Xty
    for l in 1:nl
        Ll = L[:,cols]
        trmm!('R','L','N','N',1.,Lambda,Ll)
        Ll += ee
        _, info = potrf!('L',Ll)
        if bool(info)
            error("Cholesky decomposition failed at l = $l")
        end
        for j in 1:k ldL2 += 2.log(Ll[j,j]) end
        LXZl = LXZ[:,cols]
        trmm!('R','L','N','N',1.,Lambda,LXZl)
        trsm!('R','L','T','N',1.,Ll,LXZl)
        syrk!('L','N',-1.,LXZl,1.,m.LX)
        ul = m.u[cols]
        cols += k
        trmv!('L','T','N',Lambda,ul)
        ccall(("dtrsv_",Base.LinAlg.BLAS.libblas), Void,
              (Ptr{Uint8}, Ptr{Uint8}, Ptr{Uint8}, Ptr{Base.LinAlg.BlasInt},
               Ptr{Float64}, Ptr{Base.LinAlg.BlasInt},
               Ptr{Float64}, Ptr{Base.LinAlg.BlasInt}),
              &'L', &'N', &'N', &k, Ll, &k, ul, &1)
        m.beta -= LXZl*ul
    end
    _, info = potrf!('L',m.LX)
    if bool(info) error("Downdated X'X is not positive definite") end
    potrs!('L',m.LX,m.beta)
end
