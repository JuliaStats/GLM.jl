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
mutable struct DensePredQR{T<:BlasReal, Q<:Union{QRCompactWY, QRPivoted}, W<:AbstractWeights} <: DensePred
    X::Matrix{T}                  # model matrix
    beta0::Vector{T}              # base coefficient vector
    delbeta::Vector{T}            # coefficient increment
    scratchbeta::Vector{T}
    qr::Q
    wts::W
    scratchm1::Matrix{T}

    function DensePredQR(X::AbstractMatrix, pivot::Bool, wts::W) where W<:Union{AbstractWeights}
        n, p = size(X)
        T = typeof(float(zero(eltype(X))))
        Q = pivot ? QRPivoted : QRCompactWY
        fX = float(X)
        if wts isa UnitWeights
            cfX = fX === X ? copy(fX) : fX
        else
            cfX = Diagonal(sqrt.(wts))*fX 
        end
        F = pivot ? pivoted_qr!(cfX) : qr!(cfX)
        new{T,Q,W}(Matrix{T}(X),
            zeros(T, p),
            zeros(T, p),
            zeros(T, p),
            F,
            wts,
            similar(X, T))
    end
end

DensePredQR(X::AbstractMatrix) = DensePredQR(X, false, uweights(size(X,1)))
"""
    delbeta!(p::LinPred, r::Vector)

Evaluate and return `p.delbeta` the increment to the coefficient vector from residual `r`
"""
function delbeta! end

function delbeta!(p::DensePredQR{T, <:QRCompactWY,<:AbstractWeights}, r::Vector{T}) where T<:BlasReal
    r̃ = p.wts isa UnitWeights ? r : (wtsqrt = sqrt.(p.wts); wtsqrt .*= r; wtsqrt)
    #rnk = rank(p.qr.R)
    #rnk == length(p.delbeta) || throw(RankDeficientException(rnk))
    p.delbeta = p.qr\r̃
    #mul!(p.scratchm1, Diagonal(ones(size(r))), p.X)
    return p
end

function delbeta!(p::DensePredQR{T, <:QRCompactWY,<:AbstractWeights}, r::Vector{T}, wt::AbstractVector) where T<:BlasReal
    X = p.X
    wtsqrt = sqrt.(wt)
    sqrtW = Diagonal(wtsqrt)
    mul!(p.scratchm1, sqrtW, X)
    ỹ = (wtsqrt .*= r) # to reuse wtsqrt's memory
    p.qr = qr!(p.scratchm1)
    p.delbeta = p.qr \ ỹ
    return p
end

function delbeta!(p::DensePredQR{T,<:QRPivoted,<:AbstractWeights}, r::Vector{T}) where T<:BlasReal
    r̃ = p.wts isa UnitWeights ? r : (wtsqrt = sqrt.(p.wts); wtsqrt .*= r; wtsqrt)
    rnk = rank(p.qr.R)
    if rnk == length(p.delbeta)
        p.delbeta = p.qr \ r̃
    else
        R = UpperTriangular(view(parent(p.qr.R), 1:rnk, 1:rnk))
        piv = p.qr.p
        fill!(p.delbeta, 0)
        p.delbeta[1:rnk] = R \ view(p.qr.Q'*r̃, 1:rnk)
        invpermute!(p.delbeta, piv)
    end
    return p
end

function delbeta!(p::DensePredQR{T,<:QRPivoted,<:AbstractWeights}, r::Vector{T}, wt::AbstractVector{T}) where T<:BlasReal
    X = p.X
    wtsqrt = sqrt.(wt)
    sqrtW = Diagonal(wtsqrt)
    mul!(p.scratchm1, sqrtW, X)
    r̃ = (wtsqrt .*= r) # to reuse wtsqrt's memory

    p.qr = pivoted_qr!(p.scratchm1)
    rnk = rank(p.qr.R) # FIXME! Don't use svd for this
    R = UpperTriangular(view(parent(p.qr.R), 1:rnk, 1:rnk))
    permute!(p.delbeta, p.qr.p)
    for k = (rnk + 1):length(p.delbeta)
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
- `scratchm2`: scratch Matrix{T} of the same size as `X'X`
"""
mutable struct DensePredChol{T<:BlasReal,C,W<:AbstractWeights} <: DensePred
    X::Matrix{T}                   # model matrix
    beta0::Vector{T}               # base vector for coefficients
    delbeta::Vector{T}             # coefficient increment
    scratchbeta::Vector{T}
    chol::C
    wts::W
    scratchm1::Matrix{T}
    scratchm2::Matrix{T}
end

function DensePredChol(X::AbstractMatrix, pivot::Bool, wts::AbstractWeights)
    if wts isa UnitWeights
        F = Hermitian(float(X'X))
        T = eltype(F)
        scr = similar(X, T)
    else
        T = float(promote_type(eltype(wts), eltype(X)))
        scr = similar(X, T)
        mul!(scr, Diagonal(wts), X)
        F = Hermitian(float(scr'X))
    end
    F = pivot ? pivoted_cholesky!(F, tol = -one(T), check = false) : cholesky!(F)
    DensePredChol(Matrix{T}(X),
        zeros(T, size(X, 2)),
        zeros(T, size(X, 2)),
        zeros(T, size(X, 2)),
        F,
        wts,
        scr,
        similar(cholfactors(F)))
end

DensePredChol(X::AbstractMatrix, pivot::Bool) =
    DensePredChol(X, pivot, uweights(size(X,1)))

cholpred(X::AbstractMatrix, pivot::Bool, wts::AbstractWeights=uweights(size(X,1))) = DensePredChol(X, pivot, wts)
qrpred(X::AbstractMatrix, pivot::Bool=false, wts::AbstractWeights=uweights(size(X,1))) = DensePredQR(X, pivot, wts)

cholfactors(c::Union{Cholesky,CholeskyPivoted}) = c.factors
cholesky!(p::DensePredChol{T}) where {T<:FP} = p.chol

cholesky(p::DensePredQR{T}) where {T<:FP} = Cholesky{T,typeof(p.X)}(copy(p.qr.R), 'U', 0)
function cholesky(p::DensePredChol{T}) where T<:FP
    c = p.chol
    Cholesky(copy(cholfactors(c)), c.uplo, c.info)
end

function delbeta!(p::DensePredChol{T,<:Cholesky,<:AbstractWeights}, r::Vector{T}) where T<:BlasReal
    X = p.wts isa UnitWeights ? p.scratchm1 .= p.X : mul!(p.scratchm1, Diagonal(p.wts), p.X)
    ldiv!(p.chol, mul!(p.delbeta, transpose(X), r))
    p
end

function delbeta!(p::DensePredChol{T,<:CholeskyPivoted,<:AbstractWeights}, r::Vector{T}) where T<:BlasReal
    ch = p.chol
    X = p.wts isa UnitWeights ? p.scratchm1 .= p.X : mul!(p.scratchm1, Diagonal(p.wts), p.X)
    delbeta = mul!(p.delbeta, adjoint(X), r)
    rnk = rank(ch)
    if rnk == length(delbeta)
        ldiv!(ch, delbeta)
    else
        permute!(delbeta, ch.p)
        for k=(rnk+1):length(delbeta)
            delbeta[k] = zero(T)
        end
        LAPACK.potrs!(ch.uplo, view(ch.factors, 1:rnk, 1:rnk), view(delbeta, 1:rnk))
        invpermute!(delbeta, ch.p)
    end
    p
end

function delbeta!(p::DensePredChol{T,<:Cholesky,<:AbstractWeights}, r::Vector{T}, wt::Vector{T}) where T<:BlasReal
    scr = mul!(p.scratchm1, Diagonal(wt), p.X)
    cholesky!(Hermitian(mul!(cholfactors(p.chol), transpose(scr), p.X), :U))
    mul!(p.delbeta, transpose(scr), r)
    ldiv!(p.chol, p.delbeta)
    p
end

function delbeta!(p::DensePredChol{T,<:CholeskyPivoted,<:AbstractWeights}, r::Vector{T}, wt::Vector{T}) where T<:BlasReal
    piv = p.chol.p # inverse vector
    delbeta = p.delbeta
    # p.scratchm1 = WX
    mul!(p.scratchm1, Diagonal(wt), p.X)
    # p.scratchm2 = X'WX
    mul!(p.scratchm2, adjoint(p.scratchm1), p.X)
    # delbeta = X'Wr
    mul!(delbeta, transpose(p.scratchm1), r)
    # calculate delbeta = (X'WX)\X'Wr
    rnk = rank(p.chol)
    if rnk == length(delbeta)
        cf = cholfactors(p.chol)
        cf .= p.scratchm2[piv, piv]
        cholesky!(Hermitian(cf, Symbol(p.chol.uplo)))
        ldiv!(p.chol, delbeta)
    else
        permute!(delbeta, piv)
        for k=(rnk+1):length(delbeta)
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
    p
end

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

function SparsePredChol(X::SparseMatrixCSC{T}, wts::AbstractVector) where T
    chol = cholesky(sparse(I, size(X, 2), size(X,2)))
    return SparsePredChol{eltype(X),typeof(X),typeof(chol), typeof(wts)}(X,
        X',
        zeros(T, size(X, 2)),
        zeros(T, size(X, 2)),
        zeros(T, size(X, 2)),
        chol,
        wts,
        similar(X))
end

cholpred(X::SparseMatrixCSC, pivot::Bool, wts::AbstractWeights=uweights(size(X,1))) =
    SparsePredChol(X, wts)

function delbeta!(p::SparsePredChol{T,M,C,<:UnitWeights}, r::Vector{T}, wt::Vector{T}) where {T,M,C}
    scr = mul!(p.scratchm1, Diagonal(wt), p.X)
    XtWX = p.Xt*scr
    c = p.chol = cholesky(Symmetric{eltype(XtWX),typeof(XtWX)}(XtWX, 'L'))
    p.delbeta = c \ mul!(p.delbeta, adjoint(scr), r)
end

function delbeta!(p::SparsePredChol{T,M,C,<:AbstractWeights}, r::Vector{T}, wt::Vector{T}) where {T,M,C}
    scr = mul!(p.scratchm1, Diagonal(wt.*p.wts), p.X)
    XtWX = p.Xt*scr
    c = p.chol = cholesky(Symmetric{eltype(XtWX),typeof(XtWX)}(XtWX, 'L'))
    p.delbeta = c \ mul!(p.delbeta, adjoint(scr), r)
end

function delbeta!(p::SparsePredChol{T,M,C,<:UnitWeights}, r::Vector{T}) where {T,M,C}
    scr = p.X
    XtWX = p.Xt*scr
    c = p.chol = cholesky(Symmetric{eltype(XtWX),typeof(XtWX)}(XtWX, 'L'))
    p.delbeta = c \ mul!(p.delbeta, adjoint(scr), r)
end

function delbeta!(p::SparsePredChol{T,M,C,<:AbstractWeights}, r::Vector{T}) where {T,M,C}
    scr = p.scratchm1 .= p.X.*p.wts
    XtWX = p.Xt*scr
    c = p.chol = cholesky(Symmetric{eltype(XtWX),typeof(XtWX)}(XtWX, 'L'))
    p.delbeta = c \ mul!(p.delbeta, adjoint(scr), r)
end

LinearAlgebra.cholesky(p::SparsePredChol{T}) where {T} = copy(p.chol)
LinearAlgebra.cholesky!(p::SparsePredChol{T}) where {T} = p.chol

function invqr(p::DensePredQR{T,<: QRCompactWY, <:AbstractWeights}) where T
    Rinv = inv(p.qr.R)
    Rinv*Rinv'
end

function invqr(p::DensePredQR{T,<: QRPivoted, <:AbstractWeights}) where T
    rnk = rank(p.qr.R)
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

inverse(x::DensePred) = invchol(x)
inverse(x::DensePredQR) = invqr(x)
inverse(x::SparsePredChol) = invchol(x)

working_residuals(x::LinPredModel) = x.rr.wrkresid
working_weights(x::LinPredModel) = x.rr.wrkwt

function vcov(x::LinPredModel)
    if weights(x) isa ProbabilityWeights
        ## n-1 degrees of freedom - This is coherent with the `R` package `survey`,
        ## `STATA` uses n-k
        s = nobs(x)/(nobs(x) - 1)
        mm = momentmatrix(x)
        A = invloglikhessian(x)
        _vcov(x.pp, mm, A).*s
    else
        rmul!(inverse(x.pp), dispersion(x, true))
    end
end

function _vcov(pp::DensePred, Z::Matrix, A::Matrix)
    if linpred_rank(pp) < size(Z, 2)
        nancols = [all(isnan, col) for col in eachcol(A)]
        nnancols = .!nancols
        idx, nidx = findall(nancols), findall(nnancols)
        Zv = view(Z, :, nidx)
        B = Zv'Zv
        Av = view(A, nidx, nidx)
        V = similar(pp.scratchm1, (size(A)...))
        V[nidx, nidx] = Av * B * Av
        V[idx, :] .= NaN
        V[:, idx] .= NaN
    else
        B = Z'Z
        V = A * B * A
    end
    return V
end

function _vcov(pp::SparsePredChol, Z::Matrix, A::Matrix)
    ## SparsePredChol does not handle rankdeficient cases
    B = Z'*Z
    V = A*B*A
    return V
end

function cor(x::LinPredModel)
    Σ = vcov(x)
    invstd = inv.(sqrt.(diag(Σ)))
    lmul!(Diagonal(invstd), rmul!(Σ, Diagonal(invstd)))
end

stderror(x::LinPredModel) = sqrt.(diag(vcov(x)))

function show(io::IO, obj::LinPredModel)
    println(io, nameof(typeof(obj)), '\n')
    obj.formula !== nothing && println(io, obj.formula, '\n')
    println(io, "Coefficients:\n", coeftable(obj))
end

function modelframe(f::FormulaTerm, data, contrasts::AbstractDict, ::Type{M}) where M
    Tables.istable(data) ||
        throw(ArgumentError("expected data in a Table, got $(typeof(data))"))
    t = Tables.columntable(data)
    msg = StatsModels.checknamesexist(f, t)
    msg != "" && throw(ArgumentError(msg))
    data, _ = StatsModels.missing_omit(t, f)
    sch = schema(f, data, contrasts)
    f = apply_schema(f, sch, M)
    f, modelcols(f, data)
end

modelframe(obj::LinPredModel) = obj.fr

modelmatrix(obj::LinPredModel; weighted::Bool=isweighted(obj)) = modelmatrix(obj.pp; weighted=weighted)

function modelmatrix(pp::LinPred; weighted::Bool=isweighted(pp))
    Z = if weighted
        Diagonal(sqrt.(pp.wts))*pp.X
    else
        pp.X
    end
    return Z
end

function leverage(x::LinPredModel) 
    h = vec(leverage(x.pp))
    hasfield(typeof(x.rr), :wrkwt) ? x.rr.wrkwt.*h : x.rr.wts.*h
end

function leverage(pp::DensePredChol{T,<:CholeskyPivoted}) where T
    X = modelmatrix(pp; weighted=false)
    rnk = rank(pp.chol)
    A = GLM.inverse(pp)
    p = pp.chol.p[1:rnk]
    diag(X[:,p]*A[p,p]*X[:,p]')
    # sum(x->x^2, view(X, :, p)/view(pp.chol.U, p, p), dims=2)
end

function leverage(pp::DensePredChol{T,<:Cholesky}) where T
    X = modelmatrix(pp; weighted=false)
    @show X/pp.chol.U
    sum(x -> x^2, X/pp.chol.U, dims=2)
end

function leverage(pp::DensePredQR{T,<:QRPivoted}) where T
    X = modelmatrix(pp; weighted=false)
    ch = pp.qr
    rnk = rank(ch.R)
    p = invperm(ch.p)[1:rnk]
    sum(x -> x^2, view(X, :, 1:rnk)/view(ch.R, p, p), dims=2)
end

function leverage(pp::DensePredQR{T,<:QRCompactWY}) where T
    X = modelmatrix(pp; weighted=false)
    sum(x -> x^2, X/pp.qr.R, dims=2)
end

response(obj::LinPredModel) = obj.rr.y

fitted(m::LinPredModel) = m.rr.mu
predict(mm::LinPredModel) = fitted(mm)

function StatsModels.formula(obj::LinPredModel)
    obj.formula === nothing && throw(ArgumentError("model was fitted without a formula"))
    return obj.formula
end

residuals(obj::LinPredModel; weighted::Bool=false) = residuals(obj.rr; weighted=weighted)

"""
    nobs(obj::LinearModel)
    nobs(obj::GLM)

For linear and generalized linear models, returns the number of rows, or,
when prior weights are specified, the sum of weights.
"""
nobs(obj::LinPredModel) = nobs(obj.rr)

weights(obj::RegressionModel) = weights(obj.model)
weights(m::LinPredModel) = weights(m.rr)
weights(pp::LinPred) = pp.wts

isweighted(m::LinPredModel) = isweighted(m.pp)
isweighted(pp::LinPred) = weights(pp) isa Union{FrequencyWeights, AnalyticWeights, ProbabilityWeights}

coef(x::LinPred) = x.beta0
coef(obj::LinPredModel) = coef(obj.pp)
coefnames(x::LinPredModel) =
    x.formula === nothing ? ["x$i" for i in 1:length(coef(x))] : StatsModels.vectorize(coefnames(formula(x).rhs))

dof_residual(obj::LinPredModel) = nobs(obj) - linpred_rank(obj)

hasintercept(m::LinPredModel) = any(i -> all(==(1), view(m.pp.X , :, i)), 1:size(m.pp.X, 2))

linpred_rank(x::LinPredModel) = linpred_rank(x.pp)
linpred_rank(x::LinPred) = length(x.beta0)
linpred_rank(x::DensePredChol{<:Any, <:CholeskyPivoted}) = rank(x.chol)
linpred_rank(x::DensePredChol{<:Any, <:Cholesky}) = rank(x.chol.U)
linpred_rank(x::DensePredQR{<:Any,<:QRPivoted}) = rank(x.qr.R)

ispivoted(x::LinPred) = false
ispivoted(x::DensePredChol{<:Any, <:CholeskyPivoted}) = true
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
    prediction = Tables.allocatecolumn(Union{_coltype(f.lhs), Missing}, length(nonmissings))
    fill!(prediction, missing)
    if interval === nothing
        predict!(view(prediction, nonmissings), mm, newx;
                 interval=interval, kwargs...)
        return prediction
    else
        # Finding integer indices once is faster
        nonmissinginds = findall(nonmissings)
        lower = Vector{Union{Float64, Missing}}(missing, length(nonmissings))
        upper = Vector{Union{Float64, Missing}}(missing, length(nonmissings))
        tup = (prediction=view(prediction, nonmissinginds),
               lower=view(lower, nonmissinginds),
               upper=view(upper, nonmissinginds))
        predict!(tup, mm, newx;
                 interval=interval, kwargs...)
        return (prediction=prediction, lower=lower, upper=upper)
    end
end
