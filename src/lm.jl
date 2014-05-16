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

deviance(r::LmResp) = length(r.wts) == 0 ? sumsqdiff(r.y, r.mu) : wsumsqdiff(r.wts,r.y,r.mu)
residuals(r::LmResp)= length(r.wts) == 0 ? r.y - r.mu : map(WtResid(),r.wts,r.y,r.mu)

type LmMod{T<:LinPred} <: LinPredModel
    rr::LmResp
    pp::T
end

cholfact(x::LmMod) = cholfact(x.pp)

function StatsBase.fit{LinPredT<:LinPred}(::Type{LmMod{LinPredT}}, X::Matrix, y::Vector)
    rr = LmResp(float(y)); pp = LinPredT(X)
    installbeta!(delbeta!(pp, rr.y)); updatemu!(rr, linpred(pp,0.))
    LmMod(rr, pp)
end
StatsBase.fit(::Type{LmMod}, X::Matrix, y::Vector) = StatsBase.fit(LmMod{DensePredQR}, X, y)

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
              ["x$i" for i = 1:size(mm.pp.X, 2)], 4)
end

predict(mm::LmMod, newx::Matrix) =  newx * coef(mm)

function confint(obj::LmMod, level::Real)
    hcat(coef(obj),coef(obj)) + stderr(obj) *
    quantile(TDist(df_residual(obj)), (1. - level)/2.) * [1. -1.]
end
confint(obj::LmMod) = confint(obj, 0.95)

effects(mod::RegressionModel)=(mod.model.pp.qr[:Q]'*mod.model.rr.y)[1:size(mod.model.pp.X,2)]
effects(mod::LmMod{DensePredQR{Float64}})=(mod.pp.qr[:Q]'*mod.rr.y)[1:size(mod.pp.X,2)]

type ANOVAtest
    SSH::Float64
    SSE::Float64
    MSH::Float64
    MSE::Float64
    dfH::Int
    dfE::Int
    fstat::Float64
    pval::Float64    #regular pvalue or -log10pval
    log10pval::Bool
end

#this is a test for a group of terms from a LmMod derived from a formula and dataframe
function ANOVAtest(mod::RegressionModel,terms::Array{Int,1}; log10pval=false)
    #terms is arrary of number for each term in the model to be tested together, starting with the intercept at 0
    eff=effects(mod)  #get effect for each coefficient
    ind=findin(mod.mm.assign,terms)
    eff=eff[ind]
    SSH=sum(Abs2Fun(),eff)
    dfH=length(eff)
    MSH=SSH/dfH
    SSE=deviance(mod.model.rr)
    dfE=df_residual(mod.model.pp)
    MSE=SSE/dfE
    fstat=MSH/MSE
    pval=ccdf(FDist(dfH, dfE), fstat)
    if log10pval pval= -log10(pval) end
    return ANOVAtest(SSH,SSE,MSH,MSE,dfH,dfE,fstat,pval,log10pval)
end

#this is a test for a single term from a LmMod derived from a formula and dataframe
function ANOVAtest(mod::RegressionModel,term::Int; log10pval=false)
    #term an integer for a single term in model to be tested, starting with the intercept at 0
    eff=effects(mod)  #get effect for each coefficient
    ind=findin(mod.mm.assign,term)
    eff=eff[ind]
    SSH=eff[1]*eff[1]
    dfH=length(eff)
    MSH=SSH/dfH
    SSE=deviance(mod.model.rr)
    dfE=df_residual(mod.model.pp)
    MSE=SSE/dfE
    fstat=MSH/MSE
    pval=ccdf(FDist(dfH, dfE), fstat)
    if log10pval pval= -log10(pval) end
    return ANOVAtest(SSH,SSE,MSH,MSE,dfH,dfE,fstat,pval,log10pval)
end


#this is a test for a group of coefficients/columns of X based on a LmMod derived without a formula and dataframe
function ANOVAtest(mod::LmMod{DensePredQR{Float64}},cols::Array{Int,1}; log10pval=false)
    #cols a vector of position numbers (column number of X) to be grouped together with the intercept starting at 1
    #this does not refer the terms of a model defined by a formula if a term has >1 DF
    eff=effects(mod)  #get effect for each coefficient
    eff=eff[cols]
    SSH=sum(Abs2Fun(),eff)
    dfH=length(eff)
    MSH=SSH/dfH
    SSE=deviance(mod.rr)
    dfE=df_residual(mod.pp)
    MSE=SSE/dfE
    fstat=MSH/MSE
    pval=ccdf(FDist(dfH, dfE), fstat)
    if log10pval pval= -log10(pval) end
    return ANOVAtest(SSH,SSE,MSH,MSE,dfH,dfE,fstat,pval,log10pval)
end

#this is a test for a single coefficient/columns of X based on a LmMod derived without a formula and dataframe
function ANOVAtest(mod::LmMod{DensePredQR{Float64}},col::Int; log10pval=false)
    #col means the position number of the column of X with the intercept starting at 1
    #this does not refer the terms of a model defined by a formula if a term has >1 DF
    eff=effects(mod)  #get effect for each coefficient
    eff=eff[col]
    SSH=eff[1]*eff[1]
    dfH=length(eff)
    MSH=SSH/dfH
    SSE=deviance(mod.rr)
    dfE=df_residual(mod.pp)
    MSE=SSE/dfE
    fstat=MSH/MSE
    pval=ccdf(FDist(dfH, dfE), fstat)
    if log10pval pval= -log10(pval) end
    return ANOVAtest(SSH,SSE,MSH,MSE,dfH,dfE,fstat,pval,log10pval)
end

function Base.show(io::IO,at::ANOVAtest)
    if at.log10pval
        println("              ","DF",'\t',"SS",'\t',"MS",'\t',"F",'\t',"log10pval")
    else
        println("              ","DF",'\t',"SS",'\t',"MS",'\t',"F",'\t',"pval")
    end
    println("Hypothesis    ",round(at.dfH,3),'\t',round(at.SSH,3),'\t',round(at.MSH,3),'\t',round(at.fstat,3),'\t',at.pval)
    println("Residuals     ",round(at.dfE,3),'\t', round(at.SSE,3),'\t', round(at.MSE,3))
end

