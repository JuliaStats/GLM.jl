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
mutable struct DensePredQR{T<:BlasReal, W<:AbstractWeights} <: DensePred
    X::Matrix{T}                  # model matrix
    beta0::Vector{T}              # base coefficient vector
    delbeta::Vector{T}            # coefficient increment
    scratchbeta::Vector{T}
    qr::QRCompactWY{T}
    wts::W
    function DensePredQR{T}(X::Matrix{T}, beta0::Vector{T}, wts::W) where {T,W<:AbstractWeights}
        n, p = size(X)
        length(beta0) == p || throw(DimensionMismatch("length(β0) ≠ size(X,2)"))
        length(wts) == n || throw(DimensionMismatch("Length of weights does not match the dimension of X"))
        qrX = qr(X) 
        new{T,W}(X, beta0, zeros(T,p), zeros(T,p), qrX, wts)
    end
end
DensePredQR{T}(X::Matrix) where T = DensePredQR{eltype(X)}(X, zeros(T, size(X, 2)), uweights(size(X,1)))
convert(::Type{DensePredQR{T}}, X::Matrix{T}) where {T} = DensePredQR{T}(X)

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
- `scratchm2`: scratch Matrix{T} of the same size as `X'X`
"""
mutable struct DensePredChol{T<:BlasReal,C,W<:AbstractVector} <: DensePred
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

cholpred(X::AbstractMatrix, pivot::Bool, wts::AbstractWeights=uweights(size(X,1))) =
    DensePredChol(X, pivot, wts)

cholfactors(c::Union{Cholesky,CholeskyPivoted}) = c.factors
cholesky!(p::DensePredChol{T}) where {T<:FP} = p.chol

cholesky(p::DensePredQR{T}) where {T<:FP} = Cholesky{T,typeof(p.X)}(copy(p.qr.R), 'U', 0)
function cholesky(p::DensePredChol{T}) where T<:FP
    c = p.chol
    Cholesky(copy(cholfactors(c)), c.uplo, c.info)
end
cholesky!(p::DensePredQR{T}) where {T<:FP} = Cholesky{T,typeof(p.X)}(p.qr.R, 'U', 0)

function delbeta!(p::DensePredChol{T,<:Cholesky}, r::Vector{T}) where T<:BlasReal
    X = p.wts isa UnitWeights ? p.scratchm1 .= p.X : mul!(p.scratchm1, Diagonal(p.wts), p.X)
    ldiv!(p.chol, mul!(p.delbeta, transpose(X), r))
    p
end

function delbeta!(p::DensePredChol{T,<:CholeskyPivoted}, r::Vector{T}) where T<:BlasReal
    ch = p.chol
    X = p.wts isa UnitWeights ? p.scratchm1 .= p.X : mul!(p.scratchm1, Diagonal(p.wts), p.X)
    delbeta = mul!(p.delbeta, adjoint(X), r)
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

function delbeta!(p::DensePredChol{T,<:Cholesky,<:AbstractWeights}, r::Vector{T}, wt::Vector{T}) where T<:BlasReal
    scr = mul!(p.scratchm1, Diagonal(wt), p.X)
    cholesky!(Hermitian(mul!(cholfactors(p.chol), transpose(scr), p.X), :U))
    mul!(p.delbeta, transpose(scr), r)
    ldiv!(p.chol, p.delbeta)
    p
end

function delbeta!(p::DensePredChol{T,<:CholeskyPivoted, <:AbstractWeights}, r::Vector{T}, wt::Vector{T}) where T<:BlasReal
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
    scratchm1::M
    wts::W
end

function SparsePredChol(X::SparseMatrixCSC{T}, wts::AbstractVector) where T
    chol = cholesky(sparse(I, size(X, 2), size(X,2)))
    return SparsePredChol{eltype(X),typeof(X),typeof(chol), typeof(wts)}(X,
        X',
        zeros(T, size(X, 2)),
        zeros(T, size(X, 2)),
        zeros(T, size(X, 2)),
        chol,
        similar(X),
        wts)
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
        rmul!(invchol(x.pp), dispersion(x, true))
    end
end

function _vcov(pp::DensePredChol, Z::Matrix, A::Matrix)   
    if pp.chol isa CholeskyPivoted && rank(pp.chol) != size(A, 1)
        nancols = [all(isnan, col) for col in eachcol(A)]
        nnancols = .!nancols
        Zv = view(Z, :, nnancols)
        B = Zv'Zv
        Av = view(A, nnancols, nnancols)
        V = similar(pp.scratchm2)
        V[nnancols, nnancols] = Av * B * Av
        V[nancols, :] .= NaN
        V[:, nancols] .= NaN
    else
        B = mul!(pp.scratchm2, Z', Z)
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
        mul!(pp.scratchm1, Diagonal(sqrt.(pp.wts)), pp.X)
    else
        pp.X
    end
    return Z
end

leverage(x::LinPredModel) = leverage(x.pp)

function leverage(pp::DensePredChol{T, C, W}) where {T, C<:CholeskyPivoted, W}
    X = modelmatrix(pp; weighted=isweighted(pp))
    _, k = size(X)    
    ch = pp.chol
    rnk = rank(ch)
    p = ch.p
    idx = invperm(p)[1:rnk]
    sum(x -> x^2, view(X, :, 1:rnk)/ch.U[1:rnk, idx], dims=2)
end

function leverage(pp::DensePredChol{T, C, W}) where {T, C<:Cholesky, W}
    X = modelmatrix(pp; weighted=isweighted(pp))
    sum(x -> x^2, X/pp.chol.U, dims=2)
end

response(obj::LinPredModel) = obj.rr.y

fitted(m::LinPredModel) = m.rr.mu
predict(mm::LinPredModel) = fitted(mm)

function formula(obj::LinPredModel)
    obj.formula === nothing && throw(ArgumentError("model was fitted without a formula"))
    return obj.formula
end

residuals(obj::LinPredModel; weighted::Bool=false) = residuals(obj.rr; weighted=weighted)

nobs(obj::LinPredModel) = nobs(obj.rr)

weights(obj::RegressionModel) = weights(obj.model)
weights(m::LinPredModel) = weights(m.rr)
weights(pp::LinPred) = pp.wts

isweighted(obj::RegressionModel) = isweighted(obj.model.pp)
isweighted(m::LinPredModel) = isweighted(m.pp)
isweighted(pp::LinPred) = weights(pp) isa Union{FrequencyWeights, AnalyticWeights, ProbabilityWeights}

coef(x::LinPred) = x.beta0
coef(obj::LinPredModel) = coef(obj.pp)
coefnames(x::LinPredModel) =
    x.formula === nothing ? ["x$i" for i in 1:length(coef(x))] : coefnames(formula(x).rhs)

dof_residual(obj::LinPredModel) = nobs(obj) - dof(obj) + 1

hasintercept(m::LinPredModel) = any(i -> all(==(1), view(m.pp.X , :, i)), 1:size(m.pp.X, 2))

linpred_rank(x::LinPred) = length(x.beta0)
linpred_rank(x::DensePredChol{<:Any, <:CholeskyPivoted}) = x.chol.rank

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
