module GLMSparseArraysExt

using GLM, LinearAlgebra, SparseArrays
import GLM: AbstractWeights, UnitWeights, BlasReal, uweights
import LinearAlgebra
## QR
mutable struct SparsePredQR{T,M<:SparseMatrixCSC,F,W<:AbstractWeights} <: GLM.LinPred
    X::M                           # model matrix
    beta0::Vector{T}               # base vector for coefficients
    delbeta::Vector{T}             # coefficient increment
    scratchbeta::Vector{T}
    qr::F
    wts::W
    scratch::M
end

function SparsePredQR(X::SparseMatrixCSC{T}, wts::AbstractVector) where {T}
    # The one(float(T))* part is because of a promotion issue in SPQR.jl on Julia 1.9
    fqr = qr(sparse(one(float(T)) * I, size(X)...))
    return SparsePredQR{eltype(X),typeof(X),typeof(fqr),typeof(wts)}(
        X,
        zeros(T, size(X, 2)),
        zeros(T, size(X, 2)),
        zeros(T, size(X, 2)),
        fqr,
        wts,
        similar(X)
    )
end

function GLM.qrpred(X::SparseMatrixCSC, pivot::Bool, wts::AbstractWeights=uweights(size(X, 1)))
    SparsePredQR(X, wts)
end

function GLM.delbeta!(p::SparsePredQR{T}, r::Vector{T}, wt::Vector{T}) where T
    wtsqrt = sqrt.(wt)
    Wsqrt = Diagonal(wtsqrt)
    scr = mul!(p.scratch, Wsqrt, p.X)
    p.qr = qr(scr)
    p.delbeta = p.qr \ (Wsqrt*r)
end

function GLM.delbeta!(p::SparsePredQR{T, M, F, <:UnitWeights}, r::Vector{T}) where {T<:BlasReal, M, F}
    p.qr = qr(p.X)
    p.delbeta = p.qr \ r
end

function GLM.delbeta!(p::SparsePredQR{T, M, F, <:AbstractWeights}, r::Vector{T}) where {T<:BlasReal, M, F}
    W = Diagonal(sqrt.(p.wts))
    p.qr = qr(W*p.X)
    p.delbeta = p.qr \ (W*r)
end

function GLM.inverse(x::SparsePredQR{T}) where T
    Rinv = UpperTriangular(x.qr.R) \ Diagonal(ones(T, size(x.qr.R, 2)))
    pinv = invperm(x.qr.pcol)
    RinvRinvt = Rinv*Rinv'
    return RinvRinvt[pinv, pinv]
end


## Cholesky
mutable struct SparsePredChol{T,M<:SparseMatrixCSC,C,W<:AbstractWeights} <: GLM.LinPred
    X::M                           # model matrix
    Xt::M                          # X'
    beta0::Vector{T}               # base vector for coefficients
    delbeta::Vector{T}             # coefficient increment
    scratchbeta::Vector{T}
    chol::C
    wts::W
    scratchm1::M
end

function SparsePredChol(X::SparseMatrixCSC{T}, wts::AbstractVector) where {T}
    chol = cholesky(sparse(I, size(X, 2), size(X, 2)))
    return SparsePredChol{eltype(X),typeof(X),typeof(chol),typeof(wts)}(X,
        X',
        zeros(T, size(X, 2)),
        zeros(T, size(X, 2)),
        zeros(T, size(X, 2)),
        chol,
        wts,
        similar(X))
end

function GLM.cholpred(X::SparseMatrixCSC, pivot::Bool=false, wts::AbstractWeights=uweights(size(X, 1)))
    SparsePredChol(X, wts)
end

function GLM.delbeta!(p::SparsePredChol{T, M, C, <:AbstractWeights}, r::Vector{T}) where {T<:BlasReal, M, C}
    scr = p.scratchm1 = p.X
    XtWX = p.Xt*scr
    c = p.chol = cholesky(Symmetric{eltype(XtWX),typeof(XtWX)}(XtWX, 'L'))
    p.delbeta = c \ mul!(p.delbeta, adjoint(scr), r)
end

function GLM.delbeta!(p::SparsePredChol{T, M, C, <:AbstractWeights}, r::Vector{T}, wt::Vector{T}) where {T<:BlasReal, M, C}
    scr = mul!(p.scratchm1, Diagonal(wt), p.X)
    XtWX = p.Xt * scr
    c = cholesky!(p.chol, Hermitian(XtWX))
    p.delbeta = c\ mul!(p.delbeta, adjoint(scr), r)
end

function GLM._vcov(pp::SparsePredChol, Z::Matrix, A::Matrix)
    ## SparsePredChol does not handle rankdeficient cases
    B = Z' * Z
    V = A * B * A
    return V
end

LinearAlgebra.cholesky(p::SparsePredChol{T}) where {T} = copy(p.chol)
LinearAlgebra.cholesky!(p::SparsePredChol{T}) where {T} = p.chol

GLM.invchol(x::SparsePredChol) = cholesky!(x) \ Matrix{Float64}(I, size(x.X, 2), size(x.X, 2))

GLM.inverse(x::SparsePredChol) = GLM.invchol(x)

end
