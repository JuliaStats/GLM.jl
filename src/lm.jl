type LmResp{V<:FPStoredVector} <: ModResp  # response in a linear model
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
LmResp{V<:FPStoredVector}(y::V) = LmResp{V}(fill!(similar(y), zero(eltype(V))), similar(y, 0), similar(y, 0), y)

function updatemu!{V<:FPStoredVector}(r::LmResp{V}, linPr::V)
    n = length(linPr); length(r.y) == n || error("length(linPr) is $n, should be $(length(r.y))")
    length(r.offset) == 0 ? copy!(r.mu, linPr) : map!(Add(), r.mu, linPr, r.offset)
    deviance(r)
end
updatemu!{V<:FPStoredVector}(r::LmResp{V}, linPr) = updatemu!(r, convert(V,vec(linPr)))

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

cholfact(x::LmMod) = cholfact(x.pp)

function lm(f::Formula, df::AbstractDataFrame)
    mf = ModelFrame(f, df); mm = ModelMatrix(mf)
    rr = LmResp(float(model_response(mf))); pp = DensePredQR(mm)
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
    CoefTable(hcat(cc,se,tt,ccdf(FDist(1, df_residual(mm)), abs2(tt))),
              ["Estimate","Std.Error","t value", "Pr(>|t|)"],
              coefnames(mm.fr), 4)
end

predict(mm::LmMod, newx::Matrix) =  newx * coef(mm)

function confint(obj::LmMod, level::Real)
    hcat(coef(obj),coef(obj)) + stderr(obj) *
    quantile(TDist(df_residual(obj)), (1. - level)/2.) * [1. -1.]
end
confint(obj::LmMod) = confint(obj, 0.95)

# get effects for model terms from vector for linear models Q'y = effects matrix
effects(mod::LmMod)=(mod.pp.qr[:Q]'*mod.rr.y)[1:size(mod.pp.X.m,2)]


function anovatable(mod::LmMod)
	eff=effects(mod)
	termnames=mod.fr.terms.terms
	termnames=[:Intercept,termnames,:Residuals]
	terms=[unique(mod.pp.X.assign),-1]
	DF=zeros(Int,length(terms))
	SS=zeros(Float64,length(terms))
	MS=zeros(Float64,length(terms))
	fstat=zeros(Float64,length(terms))
	pval=zeros(Float64,length(terms))
	DF[end]=df_residual(mod.pp)
	SS[end]=deviance(mod.rr)
	MS[end]=SS[end]/DF[end]

	for i=1:(length(terms)-1)
		jj=eff[mod.pp.X.assign.==terms[i]]
		DF[i]=length(jj)
		#SS[i]=sum(Abs2(),jj)
        SS[i]=sumsq(jj)
		MS[i]=SS[i]/DF[i]
		fstat[i]=MS[i]/MS[end]
		pval[i]=ccdf(FDist(DF[i], DF[end]), fstat[i])
	end
	table=DataFrame(
		Terms = termnames[2:end],
		DF = DF[2:end],
		SS = SS[2:end],
		MS = MS[2:end],
		F = fstat[2:end],
		pval = pval[2:end]
		)
end

