type LmResp{T<:FP} <: ModResp  # response in a linear model
    mu::Vector{T}     # mean response
    offset::Vector{T} # offset added to linear predictor (may have length 0)
    wts::Vector{T}    # prior weights (may have length 0)
    y::Vector{T}      # response
    function LmResp(mu::Vector{T}, off::Vector{T}, wts::Vector{T}, y::Vector{T})
        n = length(y); length(mu) == n || error("mismatched lengths of mu and y")
        ll = length(off); ll == 0 || ll == n || error("length of offset is $ll, must be $n or 0")
        ll = length(wts); ll == 0 || ll == n || error("length of wts is $ll, must be $n or 0")
        new(mu,off,wts,y)
    end
end
LmResp{T<:FP}(y::Vector{T}) = LmResp{T}(zeros(T,length(y)), T[], T[], y)

function updatemu!{T<:FP}(r::LmResp{T}, linPr::Vector{T})
    n = length(linPr); length(r.y) == n || error("length(linPr) is $n, should be $(length(r.y))")
    length(r.offset) == 0 ? copy!(r.mu, linPr) : map!(Add(), r.mu, linPr, r.offset)
    deviance(r)
end
updatemu!{T<:FP}(r::LmResp{T}, linPr) = updatemu!(r, convert(Vector{T},vec(linPr)))

type WtResid <: Functor{3} end
evaluate{T<:FP}(::WtResid,wt::T,y::T,mu::T) = (y - mu)*sqrt(wt)
result_type{T<:FP}(::WtResid,wt::T,y::T,mu::T) = T

deviance(r::LmResp) = length(r.wts) == 0 ? sumsqdiff(r.y, r.mu) : wsumsqdiff(r.wts,r.y,r.mu)
residuals(r::LmResp)= length(r.wts) == 0 ? r.y - r.mu : map(WtResid(),r.wts,r.y,r.mu)

type LmMod <: LinPredModel
    fr::ModelFrame
    rr::LmResp
    pp::LinPred
    ff::Formula
end

function lm(f::Formula, df::AbstractDataFrame)
    mf = ModelFrame(f, df); mm = ModelMatrix(mf)
    rr = LmResp(model_response(mf)); pp = DensePredQR(mm)
    installbeta!(delbeta!(pp, rr.y)); updatemu!(rr, linpred(pp,0.))
    LmMod(mf, rr, pp, f)
end
lm(f::Expr, df::AbstractDataFrame) = lm(Formula(f), df)
lm(f::String, df::AbstractDataFrame) = lm(Formula(parse(f)[1]), df)

function lmc(f::Formula, df::AbstractDataFrame)
    mf = ModelFrame(f, df); mm = ModelMatrix(mf)
    rr = LmResp(model_response(mf)); pp = DensePredChol(mm)
    installbeta!(delbeta!(pp, rr.y)); updatemu!(rr, linpred(pp,0.))
    LmMod(mf, rr, pp, f)
end
lmc(f::Expr, df::AbstractDataFrame) = lmc(Formula(f), df)
lmc(f::String, df::AbstractDataFrame) = lmc(Formula(parse(f)[1]), df)

## scale(m) -> estimate, s, of the scale parameter
## scale(m,true) -> estimate, s^2, of the squared scale parameter
function scale(x::LmMod, sqr::Bool=false)
    ssqr = deviance(x.rr)/df_residual(x)
    sqr ? ssqr : sqrt(ssqr)
end


function coeftable(mm::LmMod)
    cc = coef(mm)
    se = stderr(mm)
    tt = cc ./ se
    DataFrame({cc, se, tt, ccdf(FDist(1, df_residual(mm)), tt .* tt)},
              ["Estimate","Std.Error","t value", "Pr(>|t|)"])
end

function predict(mm::LmMod, newx::Matrix)
    newx * coef(mm)
end

function confint(obj::LmMod, level::Real)
    cft = coeftable(obj)
    hcat(coef(obj),coef(obj)) + cft["Std.Error"] *
    quantile(TDist(df_residual(obj)), (1. - level)/2.) * [1. -1.]
end
confint(obj::LmMod) = confint(obj, 0.95)
