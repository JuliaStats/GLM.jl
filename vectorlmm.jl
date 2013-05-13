## Types representing linear mixed models with vector-valued random effects

using RDatasets, NLopt
using Base.LinAlg.BLAS: gemv!, syrk!, syrk, trmm!, trmm
using Base.LinAlg.LAPACK: potrf!, potrs!
import Base.size

type VectorLMM1{Ti<:Integer} #<: LinearMixedModel
    L::Matrix{Float64}
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
    VectorLMM1{Ti}(zeros(k,q), Xt, syrk('L','N',1.,Xt),
                   XtZ, Xt*y, Ztrv, Ztnz, ZtZ, Zty, zeros(p), lower,
                   zeros(n), float64(lower .== 0.), # initial value of theta
                   zeros(q), y, false, false)
end        
size(m::VectorLMM1) = (length(m.y), size(m.X,2), size(m.ZtX, 1), 1)
grplevels(m::VectorLMM1) = [size(m.L,3)]
L(m::VectorLMM1) = [copy(m.L)]
function objective!(m::VectorLMM1, th::Vector{Float64})
    if any(th .< m.lower) error("theta = $th violates lower bounds") end
    m.theta[:] = th; n,p,q,t = size(m); L = m.L; k = size(L,1)
    tmp = zeros(k*k); tmp[bool(vec(tril(ones(k,k))))] = th
    Lambda = reshape(tmp, (k,k)); ee = eye(k)
    for ll in 1:size(L,3)
        L[:,:,ll] = cholfact!(trmm!('R','L','T','N',1.,Lambda,
                                    trmm('L','L','N','N',1.,Lambda,
                                         m.ZtZ[:,:,ll]))+ee,:L).UL
    end
    L
end
