module GLMSparseArraysExt

using GLM, LinearAlgebra, SparseArrays

mutable struct SparsePredChol{T,M<:SparseMatrixCSC,C} <: GLM.LinPred
    X::M                           # model matrix
    Xt::M                          # X'
    beta0::Vector{T}               # base vector for coefficients
    delbeta::Vector{T}             # coefficient increment
    scratchbeta::Vector{T}
    chol::C
    scratch::M
end
function SparsePredChol(X::SparseMatrixCSC{T}) where T
    chol = cholesky(sparse(I, size(X, 2), size(X,2)))
    return SparsePredChol{eltype(X),typeof(X),typeof(chol)}(X,
        X',
        zeros(T, size(X, 2)),
        zeros(T, size(X, 2)),
        zeros(T, size(X, 2)),
        chol,
        similar(X))
end

GLM.cholpred(X::SparseMatrixCSC, pivot::Bool=false) = SparsePredChol(X)

function GLM.delbeta!(p::SparsePredChol{T}, r::Vector{T}, wt::Vector{T}) where T
    scr = mul!(p.scratch, Diagonal(wt), p.X)
    XtWX = p.Xt*scr
    c = p.chol = cholesky(Symmetric{eltype(XtWX),typeof(XtWX)}(XtWX, 'L'))
    p.delbeta = c \ mul!(p.delbeta, adjoint(scr), r)
end

function GLM.delbeta!(p::SparsePredChol{T}, r::Vector{T}) where T
    scr = p.scratch = p.X
    XtWX = p.Xt*scr
    c = p.chol = cholesky(Symmetric{eltype(XtWX),typeof(XtWX)}(XtWX, 'L'))
    p.delbeta = c \ mul!(p.delbeta, adjoint(scr), r)
end

LinearAlgebra.cholesky(p::SparsePredChol{T}) where {T} = copy(p.chol)
LinearAlgebra.cholesky!(p::SparsePredChol{T}) where {T} = p.chol

GLM.invchol(x::SparsePredChol) = cholesky!(x) \ Matrix{Float64}(I, size(x.X, 2), size(x.X, 2))

GLM.inverse(x::SparsePredChol) = GLM.invchol(x)

end