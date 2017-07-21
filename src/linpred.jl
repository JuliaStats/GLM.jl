## Return the linear predictor vector
function linpred!(out, p::LinPred, f::Real=1.)
    A_mul_B!(out, p.X, f == 0 ? p.beta0 : broadcast!(muladd, p.scratchbeta, f, p.delbeta, p.beta0))
end
linpred(p::LinPred, f::Real=1.) = linpred!(Vector{eltype(p.X)}(size(p.X, 1)), p, f)

## Install beta0 + f*delbeta as beta0 and zero out delbeta
function installbeta!(p::LinPred, f::Real=1.)
    beta0 = p.beta0
    delbeta = p.delbeta
    @inbounds for i = eachindex(beta0,delbeta)
        beta0[i] += delbeta[i]*f
        delbeta[i] = 0
    end
    p.beta0
end

type DensePredQR{T<:BlasReal} <: DensePred
    X::Matrix{T}                  # model matrix
    beta0::Vector{T}              # base coefficient vector
    delbeta::Vector{T}            # coefficient increment
    scratchbeta::Vector{T}
    qr::QRCompactWY{T}
    function (::Type{DensePredQR{T}}){T}(X::Matrix{T}, beta0::Vector{T})
        n, p = size(X)
        length(beta0) == p || throw(DimensionMismatch("length(β0) ≠ size(X,2)"))
        new{T}(X, beta0, zeros(T,p), zeros(T,p), qrfact(X))
    end
end
(::Type{DensePredQR})(X::Matrix, beta0::Vector) = DensePredQR{eltype(X)}(X, beta0)
convert{T}(::Type{DensePredQR{T}}, X::Matrix{T}) = DensePredQR{T}(X, zeros(T, size(X, 2)))

function delbeta!{T<:BlasReal}(p::DensePredQR{T}, r::Vector{T})
    p.delbeta = p.qr\r
    return p
end

type DensePredChol{T<:BlasReal,C} <: DensePred
    X::Matrix{T}                   # model matrix
    beta0::Vector{T}               # base vector for coefficients
    delbeta::Vector{T}             # coefficient increment
    scratchbeta::Vector{T}
    chol::C
    scratch::Matrix{T}
end
function DensePredChol(X::StridedMatrix)
    F = cholfact!(float(X'X))
    T = eltype(F)
    DensePredChol(AbstractMatrix{T}(X),
        zeros(T, size(X, 2)),
        zeros(T, size(X, 2)),
        zeros(T, size(X, 2)),
        F,
        similar(X, T))
end

cholpred(X::StridedMatrix) = DensePredChol(X)

cholfactors(c::Cholesky) = c.factors
Base.LinAlg.cholfact!{T<:FP}(p::DensePredChol{T}) = p.chol

if VERSION < v"0.7.0-DEV.393"
    Base.LinAlg.cholfact{T<:FP}(p::DensePredQR{T}) = Cholesky{T,typeof(p.X)}(copy(p.qr[:R]), 'U')
    function Base.LinAlg.cholfact{T<:FP}(p::DensePredChol{T})
        c = p.chol
        return Cholesky(copy(cholfactors(c)), c.uplo)
    end
    Base.LinAlg.cholfact!{T<:FP}(p::DensePredQR{T}) = Cholesky{T,typeof(p.X)}(p.qr[:R], 'U')
else
    Base.LinAlg.cholfact{T<:FP}(p::DensePredQR{T}) = Cholesky{T,typeof(p.X)}(copy(p.qr[:R]), 'U', 0)
    function Base.LinAlg.cholfact{T<:FP}(p::DensePredChol{T})
        c = p.chol
        return Cholesky(copy(cholfactors(c)), c.uplo, c.info)
    end
    Base.LinAlg.cholfact!{T<:FP}(p::DensePredQR{T}) = Cholesky{T,typeof(p.X)}(p.qr[:R], 'U', 0)
end

function delbeta!{T<:BlasReal}(p::DensePredChol{T}, r::Vector{T})
    A_ldiv_B!(p.chol, At_mul_B!(p.delbeta, p.X, r))
    p
end

function delbeta!{T<:BlasReal}(p::DensePredChol{T}, r::Vector{T}, wt::Vector{T})
    scr = scale!(p.scratch, wt, p.X)
    if VERSION < v"0.5.0-dev+4677"
        cholfact!(At_mul_B!(cholfactors(p.chol), scr, p.X), :U)
    else
        cholfact!(Hermitian(At_mul_B!(cholfactors(p.chol), scr, p.X), :U))
    end
    A_ldiv_B!(p.chol, At_mul_B!(p.delbeta, scr, r))
    p
end

type SparsePredChol{T,M<:SparseMatrixCSC,C} <: GLM.LinPred
    X::M                           # model matrix
    Xt::M                          # X'
    beta0::Vector{T}               # base vector for coefficients
    delbeta::Vector{T}             # coefficient increment
    scratchbeta::Vector{T}
    chol::C
    scratch::M
end
function SparsePredChol{T}(X::SparseMatrixCSC{T})
    chol = cholfact(speye(size(X, 2)))
    return SparsePredChol{eltype(X),typeof(X),typeof(chol)}(X,
        X',
        zeros(T, size(X, 2)),
        zeros(T, size(X, 2)),
        zeros(T, size(X, 2)),
        chol,
        similar(X))
end

cholpred(X::SparseMatrixCSC) = SparsePredChol(X)

function delbeta!{T}(p::SparsePredChol{T}, r::Vector{T}, wt::Vector{T})
    scr = scale!(p.scratch, wt, p.X)
    XtWX = p.Xt*scr
    c = p.chol = cholfact(Symmetric{eltype(XtWX),typeof(XtWX)}(XtWX, 'L'))
    p.delbeta = c\Ac_mul_B!(p.delbeta, scr, r)
end

Base.cholfact{T}(p::SparsePredChol{T}) = copy(p.chol)
Base.cholfact!{T}(p::SparsePredChol{T}) = p.chol

invchol(x::DensePred) = inv(cholfact!(x))
invchol(x::SparsePredChol) = cholfact!(x) \ eye(size(x.X, 2))
vcov(x::LinPredModel) = scale!(invchol(x.pp), dispersion(x, true))

function cor(x::LinPredModel)
    Σ = vcov(x)
    invstd = similar(Σ, size(Σ, 1))
    for i = eachindex(invstd)
        invstd[i] = 1 / sqrt(Σ[i, i])
    end
    scale!(invstd, scale!(Σ, invstd))
end

stderr(x::LinPredModel) = sqrt.(diag(vcov(x)))

function show(io::IO, obj::LinPredModel)
    println(io, "$(typeof(obj)):\n\nCoefficients:\n", coeftable(obj))
end

ModelFrame(obj::LinPredModel) = obj.fr
ModelMatrix(obj::LinPredModel) = obj.pp.X
model_response(obj::LinPredModel) = obj.rr.y

fitted(m::LinPredModel) = m.rr.mu
predict(mm::LinPredModel) = fitted(mm)
formula(obj::LinPredModel) = ModelFrame(obj).formula
residuals(obj::LinPredModel) = residuals(obj.rr)

"""
    nobs(obj::LinearModel)
    nobs(obj::GLM)

For linear and generalized linear models, returns the number of rows, or,
when prior weights are specified, the sum of weights.
"""
function nobs(obj::LinPredModel)
    if isempty(obj.rr.wts)
        oftype(sum(one(eltype(obj.rr.wts))), length(obj.rr.y))
    else
        sum(obj.rr.wts)
    end
end

coef(x::LinPred) = x.beta0
coef(obj::LinPredModel) = coef(obj.pp)

dof_residual(obj::LinPredModel) = nobs(obj) - length(coef(obj))
