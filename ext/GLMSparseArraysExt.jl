module GLMSparseArraysExt

using GLM, LinearAlgebra, SparseArrays

## QR
mutable struct SparsePredQR{T,M<:SparseMatrixCSC,F} <: GLM.LinPred
    X::M                           # model matrix
    beta0::Vector{T}               # base vector for coefficients
    delbeta::Vector{T}             # coefficient increment
    scratchbeta::Vector{T}
    qr::F
    scratch::M
end
function SparsePredQR(X::SparseMatrixCSC{T}) where {T}
    # The one(float(T))* part is because of a promotion issue in SPQR.jl on Julia 1.9
    fqr = qr(sparse(one(float(T))*I, size(X)...))
    return SparsePredQR{eltype(X),typeof(X),typeof(fqr)}(X,
                                                         zeros(T, size(X, 2)),
                                                         zeros(T, size(X, 2)),
                                                         zeros(T, size(X, 2)),
                                                         fqr,
                                                         similar(X))
end

GLM.qrpred(X::SparseMatrixCSC, pivot::Bool) = SparsePredQR(X)

function GLM.delbeta!(p::SparsePredQR{T}, r::Vector{T}, wt::Vector{T}) where {T}
    wtsqrt = sqrt.(wt)
    Wsqrt = Diagonal(wtsqrt)
    scr = mul!(p.scratch, Wsqrt, p.X)
    p.qr = qr(scr)
    return p.delbeta = p.qr \ (Wsqrt*r)
end

function GLM.delbeta!(p::SparsePredQR{T}, r::Vector{T}) where {T}
    p.qr = qr(p.X)
    return p.delbeta = p.qr \ r
end

function GLM.inverse(x::SparsePredQR{T}) where {T}
    Rinv = UpperTriangular(x.qr.R) \ Diagonal(ones(T, size(x.qr.R, 2)))
    pinv = invperm(x.qr.pcol)
    RinvRinvt = Rinv*Rinv'
    return RinvRinvt[pinv, pinv]
end

## Cholesky
mutable struct SparsePredChol{T,M<:SparseMatrixCSC,C} <: GLM.LinPred
    X::M                           # model matrix
    Xt::M                          # X'
    beta0::Vector{T}               # base vector for coefficients
    delbeta::Vector{T}             # coefficient increment
    scratchbeta::Vector{T}
    chol::C
    scratch::M
end
function SparsePredChol(X::SparseMatrixCSC{T}) where {T}
    chol = cholesky(sparse(I, size(X, 2), size(X, 2)))
    return SparsePredChol{eltype(X),typeof(X),typeof(chol)}(X,
                                                            X',
                                                            zeros(T, size(X, 2)),
                                                            zeros(T, size(X, 2)),
                                                            zeros(T, size(X, 2)),
                                                            chol,
                                                            similar(X))
end

GLM.cholpred(X::SparseMatrixCSC, pivot::Bool=false) = SparsePredChol(X)

function GLM.delbeta!(p::SparsePredChol{T}, r::Vector{T}, wt::Vector{T}) where {T}
    scr = mul!(p.scratch, Diagonal(wt), p.X)
    XtWX = p.Xt*scr
    c = p.chol = cholesky(Symmetric{eltype(XtWX),typeof(XtWX)}(XtWX, 'L'))
    return p.delbeta = c \ mul!(p.delbeta, adjoint(scr), r)
end

function GLM.delbeta!(p::SparsePredChol{T}, r::Vector{T}) where {T}
    scr = p.scratch = p.X
    XtWX = p.Xt*scr
    c = p.chol = cholesky(Symmetric{eltype(XtWX),typeof(XtWX)}(XtWX, 'L'))
    return p.delbeta = c \ mul!(p.delbeta, adjoint(scr), r)
end

LinearAlgebra.cholesky(p::SparsePredChol{T}) where {T} = copy(p.chol)
LinearAlgebra.cholesky!(p::SparsePredChol{T}) where {T} = p.chol

function GLM.invchol(x::SparsePredChol)
    return cholesky!(x) \
           Matrix{Float64}(I, size(x.X, 2), size(x.X, 2))
end

GLM.inverse(x::SparsePredChol) = GLM.invchol(x)

end
