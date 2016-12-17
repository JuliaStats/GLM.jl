type LmResp{V<:FPVector} <: ModResp  # response in a linear model
    mu::V                                  # mean response
    offset::V                              # offset added to linear predictor (may have length 0)
    wts::V                                 # prior weights (may have length 0)
    y::V                                   # response
    function LmResp(mu::V, off::V, wts::V, y::V)
        n = length(y)
        length(mu) == n || error("mismatched lengths of mu and y")
        ll = length(off)
        ll == 0 || ll == n || error("length of offset is $ll, must be $n or 0")
        ll = length(wts)
        ll == 0 || ll == n || error("length of wts is $ll, must be $n or 0")
        new(mu, off, wts, y)
    end
end
convert{V<:FPVector}(::Type{LmResp{V}}, y::V) =
    LmResp{V}(zeros(y), similar(y, 0), similar(y, 0), y)

function convert{T<:Real}(::Type{LmResp}, y::AbstractVector{T})
    yy = float(y)
    convert(LmResp{typeof(yy)}, yy)
end

function updateμ!{V<:FPVector}(r::LmResp{V}, linPr::V)
    n = length(linPr)
    length(r.y) == n || error("length(linPr) is $n, should be $(length(r.y))")
    length(r.offset) == 0 ? copy!(r.mu, linPr) : broadcast!(+, r.mu, linPr, r.offset)
    deviance(r)
end
updateμ!{V<:FPVector}(r::LmResp{V}, linPr) = updateμ!(r, convert(V, vec(linPr)))

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

type LinearModel{L<:LmResp,T<:LinPred} <: LinPredModel
    rr::L
    pp::T
end

cholfact(x::LinearModel) = cholfact(x.pp)

function StatsBase.fit!(obj::LinearModel)
    installbeta!(delbeta!(obj.pp, obj.rr.y))
    updateμ!(obj.rr, linpred(obj.pp, zero(eltype(obj.rr.y))))
    return obj
end

fit(::Type{LinearModel}, X::AbstractMatrix, y::AbstractVector) = fit!(LinearModel(LmResp(y), cholpred(X)))

lm(X, y) = fit(LinearModel, X, y)

dof(x::LinearModel) = length(coef(x)) + 1

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
    se = stderr(mm)
    tt = cc ./ se
    CoefTable(hcat(cc,se,tt,ccdf(FDist(1, dof_residual(mm)), abs2.(tt))),
              ["Estimate","Std.Error","t value", "Pr(>|t|)"],
              ["x$i" for i = 1:size(mm.pp.X, 2)], 4)
end

predict(mm::LinearModel, newx::AbstractMatrix) = newx * coef(mm)

"""
For linear models, specifying `interval_type` will return a 3-tuple with the
predicted values, the upper and the lower confidence bound. Confidence intervals
delimit the uncertainty of the estimate of the predicted values, prediction
intervals delimit the estimated bounds for any new data points from the
population.

# Arguments
* `interval_type::Symbol`: one of `:confint` or `:predint`
* `level=0.95`: the level of the confidence interval
* `alpha=1-level`: an alternative way of specifying the level of the interval
"""
function predict(mm::LinearModel, newx::AbstractMatrix, interval_type::Symbol; level = 0.95, alpha = 1 - level)
    retmean = newx * coef(mm)
    interval_type == :confint || error("only :confint is currently implemented") #:predint will be implemented

    R = cholfact!(mm.pp)[:U] #get the R matrix from the QR factorization
    residvar = (ones(size(newx,2),1) * deviance(mm)/dof_residual(mm))
    retvariance = (newx/R).^2 * residvar

    interval = quantile(TDist(dof_residual(mm)), alpha/2) * sqrt.(retvariance)
    retmean, retmean .+ interval, retmean .- interval
end


function confint(obj::LinearModel, level::Real)
    hcat(coef(obj),coef(obj)) + stderr(obj) *
    quantile(TDist(dof_residual(obj)), (1. - level)/2.) * [1. -1.]
end
confint(obj::LinearModel) = confint(obj, 0.95)
