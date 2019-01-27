"""
    LmResp

Encapsulates the response for a linear model

# Members

- `mu`: current value of the mean response vector or fitted value
- `offset`: optional offset added to the linear predictor to form `mu`
- `wts`: optional vector of prior weights
- `y`: observed response vector

Either or both `offset` and `wts` may be of length 0
"""
mutable struct LmResp{V<:FPVector} <: ModResp  # response in a linear model
    mu::V                                  # mean response
    offset::V                              # offset added to linear predictor (may have length 0)
    wts::V                                 # prior weights (may have length 0)
    y::V                                   # response
    function LmResp{V}(mu::V, off::V, wts::V, y::V) where V
        n = length(y)
        length(mu) == n || error("mismatched lengths of mu and y")
        ll = length(off)
        ll == 0 || ll == n || error("length of offset is $ll, must be $n or 0")
        ll = length(wts)
        ll == 0 || ll == n || error("length of wts is $ll, must be $n or 0")
        new{V}(mu, off, wts, y)
    end
end
LmResp(y::V) where {V<:FPVector} = LmResp{V}(fill!(similar(y), 0), similar(y, 0), similar(y, 0), y)

LmResp(y::AbstractVector{T}) where T<:Real = LmResp(float(y))

function updateμ!(r::LmResp{V}, linPr::V) where V<:FPVector
    n = length(linPr)
    length(r.y) == n || error("length(linPr) is $n, should be $(length(r.y))")
    length(r.offset) == 0 ? copyto!(r.mu, linPr) : broadcast!(+, r.mu, linPr, r.offset)
    deviance(r)
end
updateμ!(r::LmResp{V}, linPr) where {V<:FPVector} = updateμ!(r, convert(V, vec(linPr)))

function deviance(r::LmResp)
    y = r.y
    mu = r.mu
    wts = r.wts
    v = zero(eltype(y)) + zero(eltype(y)) * zero(eltype(wts))
    if isempty(wts)
        @inbounds @simd for i = eachindex(y,mu)
            v += abs2(y[i] - mu[i])
        end
    else
        @inbounds @simd for i = eachindex(y,mu,wts)
            v += abs2(y[i] - mu[i])*wts[i]
        end
    end
    v
end

function nulldeviance(r::LmResp)
    y = r.y
    m = mean(y)
    wts = r.wts
    v = zero(eltype(y))*zero(eltype(wts))
    if isempty(wts)
        @inbounds @simd for i = 1:length(y)
            v += abs2(y[i] - m)
        end
    else
        @inbounds @simd for i = 1:length(y)
            v += abs2(y[i] - m)*wts[i]
        end
    end
    v
end

function loglikelihood(r::LmResp)
    n = length(r.y)
    wts = r.wts
    sw = zero(log(one(eltype(wts))))
    for w in wts
        sw += log(w)
    end
    -n/2 * (log(2π * deviance(r)/n) + 1 - sw)
end

function nullloglikelihood(r::LmResp)
    n = length(r.y)
    wts = r.wts
    sw = zero(log(one(eltype(wts))))
    for w in wts
        sw += log(w)
    end
    -n/2 * (log(2π * nulldeviance(r)/n) + 1 - sw)
end

function residuals(r::LmResp)
    y = r.y
    mu = r.mu
    if isempty(r.wts)
        y - mu
    else
        wts = r.wts
        resid = similar(y)
        @simd for i = eachindex(resid,y,mu,wts)
            @inbounds resid[i] = (y[i] - mu[i]) * sqrt(wts[i])
        end
        resid
    end
end

"""
    LinearModel

A combination of a [`LmResp`](@ref) and a [`LinPred`](@ref)

# Members

- `rr`: a `LmResp` object
- `pp`: a `LinPred` object
"""
struct LinearModel{L<:LmResp,T<:LinPred} <: LinPredModel
    rr::L
    pp::T
end

LinearAlgebra.cholesky(x::LinearModel) = cholesky(x.pp)

function StatsBase.fit!(obj::LinearModel)
    installbeta!(delbeta!(obj.pp, obj.rr.y))
    updateμ!(obj.rr, linpred(obj.pp, zero(eltype(obj.rr.y))))
    return obj
end

function fit(::Type{LinearModel}, X::AbstractMatrix, y::AbstractVector,
             allowrankdeficient::Bool=false)
    fit!(LinearModel(LmResp(y), cholpred(X, allowrankdeficient)))
end

"""
    lm(X, y, allowrankdeficient::Bool=false)

An alias for `fit(LinearModel, X, y, allowrankdeficient)`

The arguments `X` and `y` can be a `Matrix` and a `Vector` or a `Formula` and a `DataFrame`.
"""
lm(X, y, allowrankdeficient::Bool=false) = fit(LinearModel, X, y, allowrankdeficient)

dof(x::LinearModel) = length(coef(x)) + 1

dof(obj::LinearModel{<:LmResp,<:DensePredChol{<:Real,<:CholeskyPivoted}}) = obj.pp.chol.rank + 1

"""
    deviance(obj::LinearModel)

For linear models, the deviance is equal to the residual sum of squares (RSS).
"""
deviance(obj::LinearModel) = deviance(obj.rr)

"""
    nulldeviance(obj::LinearModel)

For linear models, the deviance of the null model is equal to the total sum of squares (TSS).
"""
nulldeviance(obj::LinearModel) = nulldeviance(obj.rr)
loglikelihood(obj::LinearModel) = loglikelihood(obj.rr)
nullloglikelihood(obj::LinearModel) = nullloglikelihood(obj.rr)

r2(obj::LinearModel) = 1 - deviance(obj)/nulldeviance(obj)

function adjr2(obj::LinearModel)
    n = nobs(obj)
    # dof() includes the dispersion parameter
    p = dof(obj) - 1
    1 - (1 - r²(obj))*(n-1)/(n-p)
end

function dispersion(x::LinearModel, sqr::Bool=false)
    ssqr = deviance(x.rr)/dof_residual(x)
    return sqr ? ssqr : sqrt(ssqr)
end

function coeftable(mm::LinearModel)
    cc = coef(mm)
    se = stderror(mm)
    tt = cc ./ se
    CoefTable(hcat(cc,se,tt,ccdf.(Ref(FDist(1, dof_residual(mm))), abs2.(tt))),
              ["Estimate","Std.Error","t value", "Pr(>|t|)"],
              ["x$i" for i = 1:size(mm.pp.X, 2)], 4)
end

"""
    predict(mm::LinearModel, newx::AbstractMatrix;
            interval::Union{Symbol,Nothing} = nothing, level::Real = 0.95)

If `interval` is `nothing` (the default), return a vector with the predicted values
for model `mm` and new data `newx`.
Otherwise, return a 3-column matrix with the prediction and
the lower and upper confidence bounds for a given `level` (0.95 equates alpha = 0.05).
Valid values of `interval` are `:confidence` delimiting the  uncertainty of the
predicted relationship, and `:prediction` delimiting estimated bounds for new data points.
"""
function predict(mm::LinearModel, newx::AbstractMatrix;
                 interval::Union{Symbol,Nothing}=nothing, level::Real = 0.95)
    retmean = newx * coef(mm)
    if interval === :confint
        Base.depwarn("interval=:confint is deprecated in favor of interval=:confidence")
        interval = :confidence
    end
    if interval === nothing
        return retmean
    end
    length(mm.rr.wts) == 0 || error("prediction with confidence intervals not yet implemented for weighted regression")
    R = cholesky!(mm.pp).U #get the R matrix from the QR factorization
    residvar = (ones(size(newx,2),1) * deviance(mm)/dof_residual(mm))
    if interval == :confidence
        retvariance = (newx/R).^2 * residvar
    elseif interval == :prediction
        retvariance = (newx/R).^2 * residvar .+ deviance(mm)/dof_residual(mm)
    else
        error("only :confidence and :prediction intervals are defined")
    end
    retinterval = quantile(TDist(dof_residual(mm)), (1. - level)/2) * sqrt.(retvariance)
    (prediction = retmean, lower = retmean .+ retinterval, upper = retmean .- retinterval)
end

function confint(obj::LinearModel, level::Real)
    hcat(coef(obj),coef(obj)) + stderror(obj) *
    quantile(TDist(dof_residual(obj)), (1. - level)/2.) * [1. -1.]
end
confint(obj::LinearModel) = confint(obj, 0.95)
