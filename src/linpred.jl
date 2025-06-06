"""
    linpred!(out, p::LinPred, f::Real=1.0)

Overwrite `out` with the linear predictor from `p` with factor `f`

The effective coefficient vector, `p.scratchbeta`, is evaluated as `p.beta0 .+ f * p.delbeta`,
and `out` is updated to `p.X * p.scratchbeta`
"""
function linpred!(out, p::LinPred, f::Real=1.0)
    return mul!(out, p.X,
                iszero(f) ? p.beta0 :
                broadcast!(muladd, p.scratchbeta, f, p.delbeta, p.beta0))
end

"""
    linpred(p::LinPred, f::Real=1.0)

Return the linear predictor `p.X * (p.beta0 .+ f * p.delbeta)`
"""
linpred(p::LinPred, f::Real=1.0) = linpred!(Vector{eltype(p.X)}(undef, size(p.X, 1)), p, f)

"""
    DensePredQR

A `LinPred` type with a dense QR decomposition of `X`

# Members

- `X`: Model matrix of size `n` × `p` with `n ≥ p`.  Should be full column rank.
- `beta0`: base coefficient vector of length `p`
- `delbeta`: increment to coefficient vector, also of length `p`
- `scratchbeta`: scratch vector of length `p`, used in `linpred!` method
- `qr`: either a `QRCompactWY` or `QRPivoted` object created from `X`, with optional row weights.
- `scratchm1`: scratch Matrix{T} of the same size as `X`
"""
mutable struct DensePredQR{T<:BlasReal,Q<:Union{QRCompactWY,QRPivoted}} <: DensePred
    X::Matrix{T}                  # model matrix
    beta0::Vector{T}              # base coefficient vector
    delbeta::Vector{T}            # coefficient increment
    scratchbeta::Vector{T}
    qr::Q
    scratchm1::Matrix{T}

    function DensePredQR(X::AbstractMatrix, pivot::Bool=false)
        n, p = size(X)
        T = typeof(float(zero(eltype(X))))
        Q = pivot ? QRPivoted : QRCompactWY
        fX = float(X)
        cfX = fX === X ? copy(fX) : fX
        F = pivot ? qr!(cfX, ColumnNorm()) : qr!(cfX)
        return new{T,Q}(Matrix{T}(X),
                        zeros(T, p),
                        zeros(T, p),
                        zeros(T, p),
                        F,
                        similar(X, T))
    end
end
"""
    delbeta!(p::LinPred, r::Vector)

Evaluate and return `p.delbeta` the increment to the coefficient vector from residual `r`
"""
function delbeta! end

function delbeta!(p::DensePredQR{T,<:QRCompactWY}, r::Vector{T}) where {T<:BlasReal}
    p.delbeta = p.qr \ r
    return p
end

function delbeta!(p::DensePredQR{T,<:QRCompactWY}, r::Vector{T},
                  wt::Vector{T}) where {T<:BlasReal}
    X = p.X
    wtsqrt = sqrt.(wt)
    sqrtW = Diagonal(wtsqrt)
    mul!(p.scratchm1, sqrtW, X)
    ỹ = (wtsqrt .*= r) # to reuse wtsqrt's memory
    p.qr = qr!(p.scratchm1)
    p.delbeta = p.qr \ ỹ
    return p
end

function delbeta!(p::DensePredQR{T,<:QRPivoted}, r::Vector{T}) where {T<:BlasReal}
    rnk = linpred_rank(p)
    if rnk == length(p.delbeta)
        p.delbeta = p.qr \ r
    else
        R = UpperTriangular(view(parent(p.qr.R), 1:rnk, 1:rnk))
        piv = p.qr.p
        fill!(p.delbeta, 0)
        p.delbeta[1:rnk] = R \ view(p.qr.Q'r, 1:rnk)
        invpermute!(p.delbeta, piv)
    end
    return p
end

function delbeta!(p::DensePredQR{T,<:QRPivoted}, r::Vector{T},
                  wt::Vector{T}) where {T<:BlasReal}
    X = p.X
    W = Diagonal(wt)
    wtsqrt = sqrt.(wt)
    sqrtW = Diagonal(wtsqrt)
    mul!(p.scratchm1, sqrtW, X)
    r̃ = (wtsqrt .*= r) # to reuse wtsqrt's memory

    p.qr = qr!(p.scratchm1, ColumnNorm())
    rnk = linpred_rank(p)
    R = UpperTriangular(view(parent(p.qr.R), 1:rnk, 1:rnk))
    permute!(p.delbeta, p.qr.p)
    for k in (rnk + 1):length(p.delbeta)
        p.delbeta[k] = zero(T)
    end
    p.delbeta[1:rnk] = R \ view(p.qr.Q'*r̃, 1:rnk)
    invpermute!(p.delbeta, p.qr.p)

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
mutable struct DensePredChol{T<:BlasReal,C} <: DensePred
    X::Matrix{T}                   # model matrix
    beta0::Vector{T}               # base vector for coefficients
    delbeta::Vector{T}             # coefficient increment
    scratchbeta::Vector{T}
    chol::C
    scratchm1::Matrix{T}
    scratchm2::Matrix{T}
end
function DensePredChol(X::AbstractMatrix, pivot::Bool)
    F = Hermitian(float(X'X))
    T = eltype(F)
    F = pivot ? cholesky!(F, RowMaximum(); tol=(-one(T)), check=false) : cholesky!(F)
    return DensePredChol(Matrix{T}(X),
                         zeros(T, size(X, 2)),
                         zeros(T, size(X, 2)),
                         zeros(T, size(X, 2)),
                         F,
                         similar(X, T),
                         similar(cholfactors(F)))
end

cholpred(X::AbstractMatrix, pivot::Bool=false) = DensePredChol(X, pivot)
qrpred(X::AbstractMatrix, pivot::Bool=false) = DensePredQR(X, pivot)

cholfactors(c::Union{Cholesky,CholeskyPivoted}) = c.factors
cholesky!(p::DensePredChol{T}) where {T<:FP} = p.chol

cholesky(p::DensePredQR{T}) where {T<:FP} = Cholesky{T,typeof(p.X)}(copy(p.qr.R), 'U', 0)
function cholesky(p::DensePredChol{T}) where {T<:FP}
    c = p.chol
    return Cholesky(copy(cholfactors(c)), c.uplo, c.info)
end

function delbeta!(p::DensePredChol{T,<:Cholesky}, r::Vector{T}) where {T<:BlasReal}
    ldiv!(p.chol, mul!(p.delbeta, transpose(p.X), r))
    return p
end

function delbeta!(p::DensePredChol{T,<:CholeskyPivoted}, r::Vector{T}) where {T<:BlasReal}
    ch = p.chol
    delbeta = mul!(p.delbeta, adjoint(p.X), r)
    rnk = linpred_rank(p)
    if rnk == length(delbeta)
        ldiv!(ch, delbeta)
    else
        permute!(delbeta, ch.p)
        for k in (rnk + 1):length(delbeta)
            delbeta[k] = zero(T)
        end
        LAPACK.potrs!(ch.uplo, view(ch.factors, 1:rnk, 1:rnk), view(delbeta, 1:rnk))
        invpermute!(delbeta, ch.p)
    end
    return p
end

function delbeta!(p::DensePredChol{T,<:Cholesky}, r::Vector{T},
                  wt::Vector{T}) where {T<:BlasReal}
    scr = mul!(p.scratchm1, Diagonal(wt), p.X)
    cholesky!(Hermitian(mul!(cholfactors(p.chol), transpose(scr), p.X), :U))
    mul!(p.delbeta, transpose(scr), r)
    ldiv!(p.chol, p.delbeta)
    return p
end

function delbeta!(p::DensePredChol{T,<:CholeskyPivoted}, r::Vector{T},
                  wt::Vector{T}) where {T<:BlasReal}
    piv = p.chol.p # inverse vector
    delbeta = p.delbeta
    # p.scratchm1 = WX
    mul!(p.scratchm1, Diagonal(wt), p.X)
    # p.scratchm2 = X'WX
    mul!(p.scratchm2, adjoint(p.scratchm1), p.X)
    # delbeta = X'Wr
    mul!(delbeta, transpose(p.scratchm1), r)
    # calculate delbeta = (X'WX)\X'Wr
    rnk = linpred_rank(p)
    if rnk == length(delbeta)
        cf = cholfactors(p.chol)
        cf .= p.scratchm2[piv, piv]
        cholesky!(Hermitian(cf, Symbol(p.chol.uplo)))
        ldiv!(p.chol, delbeta)
    else
        permute!(delbeta, piv)
        for k in (rnk + 1):length(delbeta)
            delbeta[k] = -zero(T)
        end
        # shift full rank column to 1:rank
        cf = cholfactors(p.chol)
        cf .= p.scratchm2[piv, piv]
        cholesky!(Hermitian(view(cf, 1:rnk, 1:rnk), Symbol(p.chol.uplo)))
        ldiv!(Cholesky(view(cf, 1:rnk, 1:rnk), Symbol(p.chol.uplo), p.chol.info),
              view(delbeta, 1:rnk))
        invpermute!(delbeta, piv)
    end
    return p
end

function invqr(p::DensePredQR{T,<: QRCompactWY}) where {T}
    Rinv = inv(p.qr.R)
    return Rinv*Rinv'
end

function invqr(p::DensePredQR{T,<: QRPivoted}) where {T}
    rnk = linpred_rank(p)
    k = length(p.delbeta)
    if rnk == k
        Rinv = inv(p.qr.R)
        xinv = Rinv*Rinv'
        ipiv = invperm(p.qr.p)
        return xinv[ipiv, ipiv]
    else
        Rsub = UpperTriangular(view(p.qr.R, 1:rnk, 1:rnk))
        RsubInv = inv(Rsub)
        xinv = fill(convert(T, NaN), (k, k))
        xinv[1:rnk, 1:rnk] = RsubInv*RsubInv'
        ipiv = invperm(p.qr.p)
        return xinv[ipiv, ipiv]
    end
end

invchol(x::DensePred) = inv(cholesky!(x))

function invchol(x::DensePredChol{T,<: CholeskyPivoted}) where {T}
    ch = x.chol
    rnk = linpred_rank(x)
    p = length(x.delbeta)
    rnk == p && return inv(ch)
    fac = ch.factors
    res = fill(convert(T, NaN), size(fac))
    for j in 1:rnk, i in 1:rnk
        res[i, j] = fac[i, j]
    end
    copytri!(LAPACK.potri!(ch.uplo, view(res, 1:rnk, 1:rnk)), ch.uplo, true)
    ipiv = invperm(ch.p)
    return res[ipiv, ipiv]
end

inverse(x::DensePred) = invchol(x)
inverse(x::DensePredQR) = invqr(x)

vcov(x::LinPredModel) = rmul!(inverse(x.pp), dispersion(x, true))

function cor(x::LinPredModel)
    Σ = vcov(x)
    invstd = inv.(sqrt.(diag(Σ)))
    return lmul!(Diagonal(invstd), rmul!(Σ, Diagonal(invstd)))
end

stderror(x::LinPredModel) = sqrt.(diag(vcov(x)))

function show(io::IO, obj::LinPredModel)
    println(io, nameof(typeof(obj)), '\n')
    obj.formula !== nothing && println(io, obj.formula, '\n')
    return println(io, "Coefficients:\n", coeftable(obj))
end

function modelframe(f::FormulaTerm, data, contrasts::AbstractDict, ::Type{M}) where {M}
    Tables.istable(data) ||
        throw(ArgumentError("expected data in a Table, got $(typeof(data))"))
    t = Tables.columntable(data)
    msg = StatsModels.checknamesexist(f, t)
    msg != "" && throw(ArgumentError(msg))
    data, _ = StatsModels.missing_omit(t, f)
    sch = schema(f, data, contrasts)
    f = apply_schema(f, sch, M)
    return f, modelcols(f, data)
end

modelmatrix(obj::LinPredModel) = obj.pp.X
response(obj::LinPredModel) = obj.rr.y

fitted(m::LinPredModel) = m.rr.mu
predict(mm::LinPredModel) = fitted(mm)
residuals(obj::LinPredModel) = residuals(obj.rr)

function StatsModels.formula(obj::LinPredModel)
    obj.formula === nothing && throw(ArgumentError("model was fitted without a formula"))
    return obj.formula
end

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
function coefnames(x::LinPredModel)
    return x.formula === nothing ? ["x$i" for i in 1:length(coef(x))] :
           coefnames(formula(x).rhs)
end

dof_residual(obj::LinPredModel) = nobs(obj) - linpred_rank(obj)

hasintercept(m::LinPredModel) = any(i -> all(==(1), view(m.pp.X, :, i)), 1:size(m.pp.X, 2))

linpred_rank(x::LinPredModel) = linpred_rank(x.pp)
linpred_rank(x::LinPred) = length(x.beta0)
linpred_rank(x::DensePredChol{<:Any,<:CholeskyPivoted}) = rank(x.chol)
linpred_rank(x::DensePredChol{<:Any,<:Cholesky}) = rank(x.chol.U)
function linpred_rank(x::DensePredQR{T,<:QRPivoted}) where {T}
    return rank(x.qr.R;
                rtol=size(x.X, 1) * eps(T))
end

ispivoted(x::LinPred) = false
ispivoted(x::DensePredChol{<:Any,<:CholeskyPivoted}) = true
ispivoted(x::DensePredQR{<:Any,<:QRPivoted}) = true

decomposition_method(x::LinPred) = isa(x, DensePredQR) ? :qr : :cholesky

_coltype(::ContinuousTerm{T}) where {T} = T

# Function common to all LinPred models, but documented separately
# for LinearModel and GeneralizedLinearModel
function StatsBase.predict(mm::LinPredModel, data;
                           interval::Union{Symbol,Nothing}=nothing,
                           kwargs...)
    Tables.istable(data) ||
        throw(ArgumentError("expected data in a Table, got $(typeof(data))"))

    f = formula(mm)
    t = Tables.columntable(data)
    cols, nonmissings = StatsModels.missing_omit(t, f.rhs)
    newx = modelcols(f.rhs, cols)
    prediction = Tables.allocatecolumn(Union{_coltype(f.lhs),Missing}, length(nonmissings))
    fill!(prediction, missing)
    if interval === nothing
        predict!(view(prediction, nonmissings), mm, newx;
                 interval=interval, kwargs...)
        return prediction
    else
        # Finding integer indices once is faster
        nonmissinginds = findall(nonmissings)
        lower = Vector{Union{Float64,Missing}}(missing, length(nonmissings))
        upper = Vector{Union{Float64,Missing}}(missing, length(nonmissings))
        tup = (prediction=view(prediction, nonmissinginds),
               lower=view(lower, nonmissinginds),
               upper=view(upper, nonmissinginds))
        predict!(tup, mm, newx;
                 interval=interval, kwargs...)
        return (prediction=prediction, lower=lower, upper=upper)
    end
end
