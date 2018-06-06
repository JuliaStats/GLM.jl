"""
    linpred!(out, p::LinPred, f::Real=1.0)

Overwrite `out` with the linear predictor from `p` with factor `f`

The effective coefficient vector, `p.scratchbeta`, is evaluated as `p.beta0 .+ f * p.delbeta`,
and `out` is updated to `p.X * p.scratchbeta`
"""
function linpred!(out, p::LinPred, f::Real=1.)
    A_mul_B!(out, p.X, iszero(f) ? p.beta0 : broadcast!(muladd, p.scratchbeta, f, p.delbeta, p.beta0))
end

"""
    linpred(p::LinPred, f::Read=1.0)

Return the linear predictor `p.X * (p.beta0 .+ f * p.delbeta)`
"""
linpred(p::LinPred, f::Real=1.) = linpred!(Vector{eltype(p.X)}(size(p.X, 1)), p, f)

"""
    installbeta!(p::LinPred, f::Real=1.0)

Install `pbeta0 .+= f * p.delbeta` and zero out `p.delbeta`.  Return the updated `p.beta0`.
"""
function installbeta!(p::LinPred, f::Real=1.)
    beta0 = p.beta0
    delbeta = p.delbeta
    @inbounds for i = eachindex(beta0,delbeta)
        beta0[i] += delbeta[i]*f
        delbeta[i] = 0
    end
    beta0
end

"""
    DensePredQR

A `LinPred` type with a dense, unpivoted QR decomposition of `X`

# Members

- `X`: Model matrix of size `n` × `p` with `n ≥ p`.  Should be full column rank.
- `beta0`: base coefficient vector of length `p`
- `delbeta`: increment to coefficient vector, also of length `p`
- `scratchbeta`: scratch vector of length `p`, used in [`linpred!`](@ref) method
- `qr`: a `QRCompactWY` object created from `X`, with optional row weights.
"""
mutable struct DensePredQR{T<:BlasReal} <: DensePred
    X::Matrix{T}                  # model matrix
    beta0::Vector{T}              # base coefficient vector
    delbeta::Vector{T}            # coefficient increment
    scratchbeta::Vector{T}
    qr::QRCompactWY{T}
    function DensePredQR{T}(X::Matrix{T}, beta0::Vector{T}) where T
        n, p = size(X)
        length(beta0) == p || throw(DimensionMismatch("length(β0) ≠ size(X,2)"))
        new{T}(X, beta0, zeros(T,p), zeros(T,p), qrfact(X))
    end
end
DensePredQR(X::Matrix, beta0::Vector) = DensePredQR{eltype(X)}(X, beta0)
convert(::Type{DensePredQR{T}}, X::Matrix{T}) where {T} = DensePredQR{T}(X, zeros(T, size(X, 2)))

"""
    delbeta!(p::LinPred, r::Vector)

Evaluate and return `p.delbeta` the increment to the coefficient vector from residual `r`
"""
function delbeta! end

function delbeta!(p::DensePredQR{T}, r::Vector{T}) where T<:BlasReal
    p.delbeta = p.qr\r
    return p
end

"""
    DensePredChol{T}

A `LinPred` type with a dense Cholesky factorization of `X'X`

# Members

- `X`: model matrix of size `n` × `p` with `n ≥ p`.  Should be full column rank.
- `beta0`: base coefficient vector of length `p`
- `delbeta`: increment to coefficient vector, also of length `p`
- `scratchbeta`: scratch vector of length `p`, used in [`linpred!`](@ref) method
- `chol`: a `Base.LinAlg.Cholesky` object created from `X'X`, possibly using row weights.
- `scratchm1`: scratch Matrix{T} of the same size as `X`
- `scratchm2`: scratch Matrix{T} os the same size as `X'X`
"""
mutable struct DensePredChol{T<:BlasReal,C} <: DensePred
    X::Matrix{T}                   # model matrix
    beta0::Vector{T}               # base vector for coefficients
    delbeta::Vector{T}             # coefficient increment
    scratchbeta::Vector{T}
    chol::C
    scratchm1::Matrix{T}
    scratchm2::Matrix{T}
end
function DensePredChol(X::StridedMatrix, pivot::Bool)
    F = Hermitian(float(X'X))
    T = eltype(F)
    F = pivot ? cholfact!(F, Val{true}, tol = -one(T)) : cholfact!(F)
    DensePredChol(AbstractMatrix{T}(X),
        zeros(T, size(X, 2)),
        zeros(T, size(X, 2)),
        zeros(T, size(X, 2)),
        F,
        similar(X, T),
        similar(cholfactors(F)))
end

cholpred(X::StridedMatrix, pivot::Bool=false) = DensePredChol(X, pivot)

cholfactors(c::Union{Cholesky,CholeskyPivoted}) = c.factors
Base.LinAlg.cholfact!(p::DensePredChol{T}) where {T<:FP} = p.chol

if VERSION < v"0.7.0-DEV.393"
    Base.LinAlg.cholfact(p::DensePredQR{T}) where {T<:FP} = Cholesky{T,typeof(p.X)}(copy(p.qr[:R]), 'U')
    function Base.LinAlg.cholfact(p::DensePredChol{T}) where T<:FP
        c = p.chol
        return Cholesky(copy(cholfactors(c)), c.uplo)
    end
    Base.LinAlg.cholfact!(p::DensePredQR{T}) where {T<:FP} = Cholesky{T,typeof(p.X)}(p.qr[:R], 'U')
else
    Base.LinAlg.cholfact(p::DensePredQR{T}) where {T<:FP} = Cholesky{T,typeof(p.X)}(copy(p.qr[:R]), 'U', 0)
    function Base.LinAlg.cholfact(p::DensePredChol{T}) where T<:FP
        c = p.chol
        return Cholesky(copy(cholfactors(c)), c.uplo, c.info)
    end
    Base.LinAlg.cholfact!(p::DensePredQR{T}) where {T<:FP} = Cholesky{T,typeof(p.X)}(p.qr[:R], 'U', 0)
end

function delbeta!(p::DensePredChol{T,<:Cholesky}, r::Vector{T}) where T<:BlasReal
    A_ldiv_B!(p.chol, At_mul_B!(p.delbeta, p.X, r))
    p
end

function delbeta!(p::DensePredChol{T,<:CholeskyPivoted}, r::Vector{T}) where T<:BlasReal
    ch = p.chol
    delbeta = Ac_mul_B!(p.delbeta, p.X, r)
    rnk = rank(ch)
    if rnk == length(delbeta)
        A_ldiv_B!(ch, delbeta)
    else
        permute!(delbeta, ch.piv)
        for k=(rnk+1):length(delbeta)
            delbeta[k] = -zero(T)
        end
        LAPACK.potrs!(ch.uplo, view(ch.factors, 1:rnk, 1:rnk), view(delbeta, 1:rnk))
        ipermute!(delbeta, ch.piv)
    end
    p
end

function delbeta!(p::DensePredChol{T,<:Cholesky}, r::Vector{T}, wt::Vector{T}) where T<:BlasReal
    scr = scale!(p.scratchm1, wt, p.X)
    cholfact!(Hermitian(At_mul_B!(cholfactors(p.chol), scr, p.X), :U))
    A_ldiv_B!(p.chol, At_mul_B!(p.delbeta, scr, r))
    p
end

function delbeta!(p::DensePredChol{T,<:CholeskyPivoted}, r::Vector{T}, wt::Vector{T}) where T<:BlasReal
    cf = cholfactors(p.chol)
    piv = p.chol.piv
    cf .= Ac_mul_B!(p.scratchm2, scale!(p.scratchm1, wt, p.X), p.X)[piv, piv]
    cholfact!(Hermitian(cf, Symbol(p.chol.uplo)))
    A_ldiv_B!(p.chol, At_mul_B!(p.delbeta, p.scratchm1, r))
    p
end

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

function delbeta!(p::SparsePredChol{T}, r::Vector{T}, wt::Vector{T}) where T
    scr = scale!(p.scratch, wt, p.X)
    XtWX = p.Xt*scr
    c = p.chol = cholfact(Symmetric{eltype(XtWX),typeof(XtWX)}(XtWX, 'L'))
    p.delbeta = c\Ac_mul_B!(p.delbeta, scr, r)
end

Base.cholfact(p::SparsePredChol{T}) where {T} = copy(p.chol)
Base.cholfact!(p::SparsePredChol{T}) where {T} = p.chol

invchol(x::DensePred) = inv(cholfact!(x))
function invchol(x::DensePredChol{T,<: CholeskyPivoted}) where T
    ch = x.chol
    rnk = rank(ch)
    p = length(x.delbeta)
    rnk == p && return inv(ch)
    fac = ch.factors
    res = fill(convert(T, NaN), size(fac))
    for j in 1:rnk, i in 1:rnk
        res[i, j] = fac[i, j]
    end
    copytri!(LAPACK.potri!(ch.uplo, view(res, 1:rnk, 1:rnk)), ch.uplo, true)
    ipiv = invperm(ch.piv)
    res[ipiv, ipiv]
end
invchol(x::SparsePredChol) = cholfact!(x) \ eye(printlnsize(x.X, 2))
vcov(x::LinPredModel) = scale!(invchol(x.pp), dispersion(x, true))

function cor(x::LinPredModel)
    Σ = vcov(x)
    invstd = similar(Σ, size(Σ, 1))
    for i = eachindex(invstd)
        invstd[i] = 1 / sqrt(Σ[i, i])
    end
    scale!(invstd, scale!(Σ, invstd))
end

stderror(x::LinPredModel) = sqrt.(diag(vcov(x)))

function show(io::IO, obj::LinPredModel)
    println(io, "$(typeof(obj)):\n\nCoefficients:\n", coeftable(obj))
end

modelframe(obj::LinPredModel) = obj.fr
modelmatrix(obj::LinPredModel) = obj.pp.X
model_response(obj::LinPredModel) = obj.rr.y

fitted(m::LinPredModel) = m.rr.mu
predict(mm::LinPredModel) = fitted(mm)
formula(obj::LinPredModel) = modelframe(obj).formula
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
