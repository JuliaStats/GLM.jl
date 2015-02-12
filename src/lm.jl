type LmResp{V<:FPVector} <: ModResp  # response in a linear model
    mu::V                                  # mean response
    offset::V                              # offset added to linear predictor (may have length 0)
    wts::V                                 # prior weights (may have length 0)
    y::V                                   # response
    function LmResp(mu::V, off::V, wts::V, y::V)
        n = length(y); length(mu) == n || error("mismatched lengths of mu and y")
        ll = length(off); ll == 0 || ll == n || error("length of offset is $ll, must be $n or 0")
        ll = length(wts); ll == 0 || ll == n || error("length of wts is $ll, must be $n or 0")
        new(mu,off,wts,y)
    end
end
LmResp{V<:FPVector}(y::V) = LmResp{V}(fill!(similar(y), zero(eltype(V))), similar(y, 0), similar(y, 0), y)

function updatemu!{V<:FPVector}(r::LmResp{V}, linPr::V)
    n = length(linPr); length(r.y) == n || error("length(linPr) is $n, should be $(length(r.y))")
    length(r.offset) == 0 ? copy!(r.mu, linPr) : map!(Add(), r.mu, linPr, r.offset)
    deviance(r)
end
updatemu!{V<:FPVector}(r::LmResp{V}, linPr) = updatemu!(r, convert(V,vec(linPr)))

type WtResid <: Functor{3} end
evaluate{T<:FP}(::WtResid,wt::T,y::T,mu::T) = (y - mu)*sqrt(wt)
result_type{T<:FP}(::WtResid,wt::T,y::T,mu::T) = T

function deviance(r::LmResp)
    y = r.y
    mu = r.mu
    wts = r.wts
    v = zero(eltype(y))+zero(eltype(y))*zero(eltype(wts))
    if isempty(wts)
        @inbounds @simd for i = 1:length(y)
            v += abs2(y[i] - mu[i])
        end
    else
        @inbounds @simd for i = 1:length(y)
            v += abs2(y[i] - mu[i])*wts[i]
        end
    end
    v
end
residuals(r::LmResp)= length(r.wts) == 0 ? r.y - r.mu : map(WtResid(),r.wts,r.y,r.mu)

type LinearModel{T<:LinPred} <: LinPredModel
    rr::LmResp
    pp::T
end

cholfact(x::LinearModel) = cholfact(x.pp)

function StatsBase.fit{LinPredT<:LinPred}(::Type{LinearModel{LinPredT}}, X::Matrix, y::Vector)
    rr = LmResp(float(y)); pp = LinPredT(X)
    installbeta!(delbeta!(pp, rr.y)); updatemu!(rr, linpred(pp,0.))
    LinearModel(rr, pp)
end
StatsBase.fit(::Type{LinearModel}, X::Matrix, y::Vector) = StatsBase.fit(LinearModel{DensePredQR}, X, y)

lm(X, y) = fit(LinearModel, X, y)
lmc(X, y) = fit(LinearModel{DensePredChol}, X, y)

## scale(m) -> estimate, s, of the scale parameter
## scale(m,true) -> estimate, s^2, of the squared scale parameter
function scale(x::LinearModel, sqr::Bool=false)
    ssqr = deviance(x.rr)/df_residual(x)
    sqr ? ssqr : sqrt(ssqr)
end

function coeftable(mm::LinearModel)
    cc = coef(mm)
    se = stderr(mm)
    tt = cc ./ se
    CoefTable(hcat(cc,se,tt,ccdf(FDist(1, df_residual(mm)), abs2(tt))),
              ["Estimate","Std.Error","t value", "Pr(>|t|)"],
              ["x$i" for i = 1:size(mm.pp.X, 2)], 4)
end

predict(mm::LinearModel, newx::Matrix) =  newx * coef(mm)

function confint(obj::LinearModel, level::Real)
    hcat(coef(obj),coef(obj)) + stderr(obj) *
    quantile(TDist(df_residual(obj)), (1. - level)/2.) * [1. -1.]
end
confint(obj::LinearModel) = confint(obj, 0.95)
