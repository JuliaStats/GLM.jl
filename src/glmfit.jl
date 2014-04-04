type GlmResp{V<:FPVector} <: ModResp           # response in a glm model
    y::V                                       # response
    d::UnivariateDistribution
    l::Link
    devresid::V                                # (squared) deviance residuals
    eta::V                                     # linear predictor
    mu::V                                      # mean response
    mueta::V                                   # derivative of mu w.r.t. eta
    offset::V                                  # offset added to linear predictor (usually 0)
    var::V                                     # (unweighted) variance at current mu
    wts::V                                     # prior weights
    wrkwts::V                                  # working weights
    wrkresid::V                                # working residuals
    function GlmResp(y::V, d::UnivariateDistribution, l::Link,
                     eta::V, mu::V,
                     off::V, wts::V)
        if isa(d, Binomial)
            for yy in y; 0. <= yy <= 1. || error("$yy in y is not in [0,1]"); end
        else
            insupport(d, y) || error("y must be in the support of d")
        end
        n = length(y)
        length(eta) == length(mu) == length(wts) == n || error("mismatched sizes")
        lo = length(off); lo == 0 || lo == n || error("offset must have length $n or length 0")
        res = new(y,d,l,similar(y),eta,mu,similar(y),off,similar(y),wts,similar(y),similar(y))
        updatemu!(res, eta)
        res
    end
end

# returns the sum of the squared deviance residuals
deviance(r::GlmResp) = sum(r.devresid)

# update the `devresid` field
devresid!(r::GlmResp) = devresid!(r.d, r.devresid, r.y, r.mu, r.wts)

# apply the link function generating the linear predictor (eta) vector from the mean vector (mu)
linkfun!(r::GlmResp) = linkfun!(r.l, r.eta, r.mu)

# apply the inverse link function generating the mean vector (mu) from the linear predictor (eta)
linkinv!(r::GlmResp) = linkinv!(r.l, r.mu, r.eta)

# evaluate the mueta vector (derivative of mu w.r.t. eta) from the linear predictor (eta)
mueta!(r::GlmResp) = mueta!(r.l, r.mueta, r.eta)

function updatemu!{T<:FPVector}(r::GlmResp{T}, linPr::T)
    n = length(linPr)
    length(r.offset) == n ? map!(Add(), r.eta, linPr, r.offset) : copy!(r.eta, linPr)
    linkinv!(r); mueta!(r); var!(r); wrkresid!(r); devresid!(r)
    sum(r.devresid)
end

updatemu!{T<:FPVector}(r::GlmResp{T}, linPr) = updatemu!(r, convert(T,vec(linPr)))

var!(r::GlmResp) = var!(r.d, r.var, r.mu)

wrkresid!(r::GlmResp) = map1!(Divide(), map!(Subtract(), r.wrkresid, r.y, r.mu), r.mueta)

function wrkresp(r::GlmResp)
    if length(r.offset) > 0 return map1!(Add(), map(Subtract(), r.eta, r.offset), r.wrkresid) end
    map(Add(), r.eta, r.wrkresid)
end

function wrkwt!(r::GlmResp)
    wrkwts = r.wrkwts
    mueta = r.mueta
    var = r.var
    if length(r.wts) == 0
        for i = 1:length(r.var)
            @inbounds wrkwts[i] = abs2(mueta[i])/var[i]
        end
    else
        wts = r.wts
        for i = 1:length(r.var)
            @inbounds wrkwts[i] = wts[i] * abs2(mueta[i])/var[i]
        end
    end
    wrkwts
end

type GlmMod <: LinPredModel
    fr::ModelFrame
    rr::GlmResp
    pp::LinPred
    ff::Formula
    fit::Bool
end

function coeftable(mm::GlmMod)
    cc = coef(mm)
    se = stderr(mm)
    zz = cc ./ se
    CoefTable(hcat(cc,se,zz,2.0 * ccdf(Normal(), abs(zz))),
              ["Estimate","Std.Error","z value", "Pr(>|z|)"],
              coefnames(mm.fr), 4)
end

function confint(obj::GlmMod, level::Real)
    hcat(coef(obj),coef(obj)) + stderr(obj)*quantile(Normal(),(1. -level)/2.)*[1. -1.]
end
confint(obj::GlmMod) = confint(obj, 0.95)
        
deviance(m::GlmMod)  = deviance(m.rr)

function fit(m::GlmMod; verbose::Bool=false, maxIter::Integer=30, minStepFac::Real=0.001, convTol::Real=1.e-6)
    m.fit && return m
    maxIter >= 1 || error("maxIter must be positive")
    0 < minStepFac < 1 || error("minStepFac must be in (0, 1)")

    cvg = false; p = m.pp; r = m.rr
    scratch = similar(p.X.m)
    devold = updatemu!(r, linpred(delbeta!(p, wrkresp(r), wrkwt!(r), scratch)))
    installbeta!(p)
    for i=1:maxIter
        f = 1.0
        dev = updatemu!(r, linpred(delbeta!(p, r.wrkresid, wrkwt!(r), scratch)))
        while dev > devold
            f /= 2.; f > minStepFac || error("step-halving failed at beta0 = $(p.beta0)")
            dev = updatemu!(r, linpred(p, f))
        end
        installbeta!(p, f)
        crit = (devold - dev)/dev
        verbose && println("$i: $dev, $crit")
        if crit < convTol; cvg = true; break end
        devold = dev
    end
    cvg || error("failure to converge in $maxIter iterations")
    m.fit = true
    m
end

function glm(f::Formula, df::AbstractDataFrame, d::UnivariateDistribution, l::Link; dofit::Bool=true, wts=Float64[], offset=Float64[])
    mf = ModelFrame(f, df)
    mm = ModelMatrix(mf)
    y = model_response(mf); T = eltype(y);
    if T <: Integer
        y = float64(y)
        T = Float64
    end
    n = length(y); lw = length(wts)
    lw == 0 || lw == n || error("length(wts) = $lw should be 0 or $n")
    w = lw == 0 ? ones(T,n) : (T <: Float64 ? copy(wts) : convert(typeof(y), wts))
    mu = mustart(d, y, w)
    eta = linkfun!(l, similar(mu), mu)
    off = T <: Float64 ? copy(offset) : convert(Vector{T}, offset)
    if !isempty(off)
        subtract!(eta, off)
    end
    rr = GlmResp{typeof(y)}(y, d, l, eta, mu, off, w)
    res = GlmMod(mf, rr, DensePredChol(mm), f, false)
    dofit ? fit(res) : res
end

glm(e::Expr, df::AbstractDataFrame, d::UnivariateDistribution, l::Link) = glm(Formula(e),df,d,l)
glm(e::Expr, df::AbstractDataFrame, d::UnivariateDistribution) = glm(Formula(e),df,d,canonicallink(d))
glm(f::Formula, df::AbstractDataFrame, d::UnivariateDistribution) = glm(f, df, d, canonicallink(d))
glm(s::String, df::AbstractDataFrame, d::UnivariateDistribution) = glm(Formula(parse(s)[1]), df, d)

## scale(m) -> estimate, s, of the scale parameter
## scale(m,true) -> estimate, s^2, of the squared scale parameter
type DispersionFun <: Functor{2} end
evaluate{T<:FP}(::DispersionFun,wt::T,resid::T) = wt*abs2(resid)
function scale(m::GlmMod, sqr::Bool=false)
    if isa(m.rr.d, Union(Bernoulli, Poisson))
        return 1.
    end

    s = sum(DispersionFun(), m.rr.wrkwts, m.rr.wrkresid)/df_residual(m)
    sqr ? s : sqrt(s)
end
