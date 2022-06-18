"""
linpred!(out, p::LinPred, f::Real=1.0)

Overwrite `out` with the linear predictor from `p` with factor `f`

The effective coefficient vector, `p.scratchbeta`, is evaluated as `p.beta0 .+ f * p.delbeta`,
and `out` is updated to `p.X * p.scratchbeta`
"""
function linpred!(out, p::LinPred, f::Real=1.)
    mul!(out, p.X, iszero(f) ? p.beta0 : broadcast!(muladd, p.scratchbeta, f, p.delbeta, p.beta0))
end

"""
linpred(p::LinPred, f::Real=1.0)

Return the linear predictor `p.X * (p.beta0 .+ f * p.delbeta)`
"""
linpred(p::LinPred, f::Real=1.) = linpred!(Vector{eltype(p.X)}(undef, size(p.X, 1)), p, f)

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
- `scratchbeta`: scratch vector of length `p`, used in `linpred!` method
- `qr`: a `QRCompactWY` object created from `X`, with optional row weights.
"""
mutable struct DensePredQR{T<:BlasReal, W<:AbstractVector{<:Real}} <: DensePred
    X::Matrix{T}                  # model matrix
    Xw::Matrix{T}                 # weighted model matrix
    beta0::Vector{T}              # base coefficient vector
    delbeta::Vector{T}            # coefficient increment
    scratchbeta::Vector{T}
    qr::QRCompactWY{T}
    wts::W
    function DensePredQR{T}(X::Matrix{T}, beta0::Vector{T}, wts::W) where {T,W<:AbstractWeights{<:Real}}
        n, p = size(X)
        length(beta0) == p || throw(DimensionMismatch("length(β0) ≠ size(X,2)"))
        (length(wts) == n || isempty(wts)) || throw(DimensionMismatch("Lenght of weights does not match the dimension of X"))
        Xw = isempty(_wt) ? Matrix{T}(undef, 0, 0) : sqrt.(wts).*X
        qrX = isempty(_wts) ? qr(X) : qr(Xw)
        new{T,W}(X, Xw, beta0, zeros(T,p), zeros(T,p), qrX, wts)
    end
    function DensePredQR{T}(X::Matrix{T}, wts::W) where {T,W}
        n, p = size(X)
        DensePredQR(X, zeros(T, p), wts)
    end
    function DensePredQR(X::Matrix{T}) where T
        n, p = size(X)
        DensePredQR{T}(X, zeros(T, p), uweights(0))
    end
end
DensePredQR(X::Matrix, beta0::Vector, wts::AbstractVector) = DensePredQR{eltype(X)}(X, beta0, wts)
DensePredQR(X::Matrix{T}, wts::AbstractVector) where T = DensePredQR{T}(X, zeros(T, size(X,2)), wts)
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
- `scratchbeta`: scratch vector of length `p`, used in `linpred!` method
- `chol`: a `Cholesky` object created from `X'X`, possibly using row weights.
- `scratchm1`: scratch Matrix{T} of the same size as `X`
- `scratchm2`: scratch Matrix{T} os the same size as `X'X`
"""
mutable struct DensePredChol{T<:BlasReal,W<:AbstractVector{<:Real},C} <: DensePred
    X::Matrix{T}                   # model matrix
    Xw::Matrix{T}                  # weighted model matrix
    beta0::Vector{T}               # base vector for coefficients
    delbeta::Vector{T}             # coefficient increment
    scratchbeta::Vector{T}
    wts::W
    chol::C    
    scratchm1::Matrix{T}
    scratchm2::Matrix{T}
end
function DensePredChol(X::AbstractMatrix, pivot::Bool, wts::AbstractWeights{<:Real})
    Xw = isempty(wts) ? Matrix{eltype(X)}(undef, 0, 0) : sqrt.(wts).*X
    F = isempty(wts) ? Hermitian(float(X'X)) : Hermitian(float(Xw'Xw))
    T = eltype(F)
    F = pivot ? pivoted_cholesky!(F, tol = -one(T), check = false) : cholesky!(F)
    DensePredChol(Matrix{T}(X),
    Matrix{T}(Xw),
    zeros(T, size(X, 2)),
    zeros(T, size(X, 2)),
    zeros(T, size(X, 2)),
    wts,
    F,
    similar(X, T),
    similar(cholfactors(F)))
end

cholpred(X::AbstractMatrix, pivot::Bool, wts::AbstractWeights) = DensePredChol(X, pivot, wts)
cholpred(X::AbstractMatrix, pivot::Bool=false) = DensePredChol(X, pivot, uweights(0))

cholfactors(c::Union{Cholesky,CholeskyPivoted}) = c.factors
cholesky!(p::DensePredChol{T}) where {T<:FP} = p.chol

cholesky(p::DensePredQR{T}) where {T<:FP} = Cholesky{T,typeof(p.X)}(copy(p.qr.R), 'U', 0)
function cholesky(p::DensePredChol{T}) where T<:FP
    c = p.chol
    Cholesky(copy(cholfactors(c)), c.uplo, c.info)
end
cholesky!(p::DensePredQR{T}) where {T<:FP} = Cholesky{T,typeof(p.X)}(p.qr.R, 'U', 0)

function delbeta!(p::DensePredChol{T,<:Cholesky}, r::Vector{T}) where T<:BlasReal
    ldiv!(p.chol, mul!(p.delbeta, transpose(p.X), r))
    p
end

function delbeta!(p::DensePredChol{T,<:AbstractWeights,<:CholeskyPivoted}, r::Vector{T}) where T<:BlasReal
    ch = p.chol
    Z = isempty(p.wts) ? p.X : p.Xw
    delbeta = mul!(p.delbeta, adjoint(Z), r)
    rnk = rank(ch)
    if rnk == length(delbeta)
        ldiv!(ch, delbeta)
    else
        permute!(delbeta, ch.p)
        for k=(rnk+1):length(delbeta)
            delbeta[k] = -zero(T)
        end
        LAPACK.potrs!(ch.uplo, view(ch.factors, 1:rnk, 1:rnk), view(delbeta, 1:rnk))
        invpermute!(delbeta, ch.p)
    end
    p
end

function delbeta!(p::DensePredChol{T,<:AbstractWeights,<:Cholesky}, r::Vector{T}) where T<:BlasReal
    Z = isempty(p.wts) ? p.X : p.Xw
    cholesky!(Hermitian(mul!(cholfactors(p.chol), transpose(Z), Z), :U))
    mul!(p.delbeta, transpose(Z), r)
    ldiv!(p.chol, p.delbeta)
    p
end

function delbeta!(p::DensePredChol{T,<:AbstractWeights,<:CholeskyPivoted}, r::Vector{T}) where T<:BlasReal   
    Z = isempty(p.wts) ? p.X : p.Xw
    cf = cholfactors(p.chol)
    piv = p.chol.p
    cf .= mul!(p.scratchm2, adjoint(Z), Z)[piv, piv]
    cholesky!(Hermitian(cf, Symbol(p.chol.uplo)))
    ldiv!(p.chol, mul!(p.delbeta, transpose(Z), r))
    p
end

mutable struct SparsePredChol{T,W<:AbstractWeights{<:Real},M<:SparseMatrixCSC,C} <: GLM.LinPred
    X::M                           # model matrix
    Xw::M                          # weighted model matrix
    Xt::M                          # X'
    beta0::Vector{T}               # base vector for coefficients
    delbeta::Vector{T}             # coefficient increment
    scratchbeta::Vector{T}
    wts::W
    chol::C
    scratch::M
end
function SparsePredChol(X::SparseMatrixCSC{T}, wts::AbstractVector) where T
    chol = cholesky(sparse(I, size(X, 2), size(X,2)))
    sqrtwts = sqrt.(wts)
    Xw = isempty(wts) ? SparseMatrixCSC(I, 0, 0) : sqrtwts.*X
    return SparsePredChol{eltype(X),typeof(X),typeof(chol)}(X,
    Xw,
    isempty(wts) ? X' : Xw',
    zeros(T, size(X, 2)),
    zeros(T, size(X, 2)),
    zeros(T, size(X, 2)),
    chol,
    similar(X))
end

cholpred(X::SparseMatrixCSC, pivot::Bool=false, wts::AbstractVector=uweights(0)) = SparsePredChol(X, wts)

function delbeta!(p::SparsePredChol{T}, r::Vector{T}, wt::Vector{T}) where T
    Z = isempty(p.wts) ? X : Xw
    #scr = mul!(p.scratch, Diagonal(wt), Z)
    XtWX = p.Xt*Z
    c = p.chol = cholesky(Symmetric{eltype(XtWX),typeof(XtWX)}(XtWX, 'L'))
    p.delbeta = c \ mul!(p.delbeta, adjoint(Z), r)
end

function delbeta!(p::SparsePredChol{T}, r::Vector{T}) where T
    scr = p.scratch = p.X
    XtWX = p.Xt*scr
    c = p.chol = cholesky(Symmetric{eltype(XtWX),typeof(XtWX)}(XtWX, 'L'))
    p.delbeta = c \ mul!(p.delbeta, adjoint(scr), r)
end

LinearAlgebra.cholesky(p::SparsePredChol{T}) where {T} = copy(p.chol)
LinearAlgebra.cholesky!(p::SparsePredChol{T}) where {T} = p.chol

invchol(x::DensePred) = inv(cholesky!(x))
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
    ipiv = invperm(ch.p)
    res[ipiv, ipiv]
end
invchol(x::SparsePredChol) = cholesky!(x) \ Matrix{Float64}(I, size(x.X, 2), size(x.X, 2))

function vcov(x::LinPredModel) 
    d = dispersion(x, true)
    B = _covm(x.pp)
    rmul!(B, dispersion(x, true))
end

_covm(pp::DensePredChol{T, W}) where {T,W} = invchol(pp)

function _covm(pp::DensePredChol{T, <:ProbabilityWeights, <:Cholesky}) where {T} 
    wts = pp.wts
    Z = pp.scratchm1 .= pp.X.*wts
    XtW2X = Z'Z
    invXtWX = invchol(pp)
    invXtWX*XtW2X*invXtWX
end

function _covm(pp::DensePredChol{T, <:ProbabilityWeights, <:CholeskyPivoted}) where {T} 
    wts = pp.wts
    Z = pp.scratchm1 .= pp.X.*wts
    rnk = rank(pp.chol)
    p = length(pp.delbeta)
    if rnk == p
        XtW2X = Z'Z
    else
        ## no idea
    end
    invXtWX = invchol(pp)
    invXtWX*XtW2X*invXtWX
end

function cor(x::LinPredModel)
    Σ = vcov(x)
    invstd = inv.(sqrt.(diag(Σ)))
    lmul!(Diagonal(invstd), rmul!(Σ, Diagonal(invstd)))
end

stderror(x::LinPredModel) = sqrt.(diag(vcov(x)))

function show(io::IO, obj::LinPredModel)
    println(io, "$(typeof(obj)):\n\nCoefficients:\n", coeftable(obj))
end

modelframe(obj::LinPredModel) = obj.fr

function modelmatrix(obj::LinPredModel; weighted=false) 
    if !weighted
        obj.pp.X
    elseif !isempty(weights(obj))
        obj.pp.Xw
    else
        throw(ArgumentError("`weighted=true` allowed only for weighted models."))
    end
end

response(obj::LinPredModel) = obj.rr.y

fitted(m::LinPredModel) = m.rr.mu
predict(mm::LinPredModel) = fitted(mm)
StatsModels.formula(obj::LinPredModel) = modelframe(obj).formula
residuals(obj::LinPredModel; kwarg...) = residuals(obj.rr; kwarg...)
weights(obj::LinPredModel) = weights(obj.rr)

coef(x::LinPred) = x.beta0
coef(obj::LinPredModel) = coef(obj.pp)

dof_residual(obj::LinPredModel) = nobs(obj) - dof(obj) + 1

hasintercept(m::LinPredModel) = any(i -> all(==(1), view(m.pp.X , :, i)), 1:size(m.pp.X, 2))
