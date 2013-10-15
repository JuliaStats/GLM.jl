type GlmResp{T<:FP} <: ModResp               # response in a glm model
    y::Vector{T}                # response
    d::UnivariateDistribution
    l::Link
    devresid::Vector{T}         # (squared) deviance residuals
    eta::Vector{T}              # linear predictor
    mu::Vector{T}               # mean response
    mueta::Vector{T}            # derivative of mu w.r.t. eta
    offset::Vector{T}           # offset added to linear predictor (usually 0)
    var::Vector{T}              # (unweighted) variance at current mu
    wts::Vector{T}              # prior weights
    wrkresid::Vector{T}         # working residuals
    function GlmResp(y::Vector{T}, d::Distribution, l::Link, eta::Vector{T},
                     mu::Vector{T}, off::Vector{T}, wts::Vector{T})
        if isa(d, Binomial)
            for yy in y; 0. <= yy <= 1. || error("$yy in y is not in [0,1]"); end
        else
            insupport(d, y) || error("y must be in the support of d")
        end
        n = length(y)
        length(eta) == length(mu) == length(wts) == n || error("mismatched sizes")
        lo = length(off); lo == 0 || lo == n || error("offset must have length $n or length 0")
        res = new(y,d,l,Array(T,n),eta,mu,Array(T,n),off,Array(T,n),wts,Array(T,n))
        updatemu!(res, eta)
        res
    end
end

deviance(r::GlmResp) = sum(r.devresid)

devresid!(r::GlmResp) = devresid!(r.d, r.devresid, r.y, r.mu, r.wts)

linkfun!(r::GlmResp) = linkfun!(r.l, r.eta, r.mu)

linkinv!(r::GlmResp) = linkinv!(r.l, r.mu, r.eta)

mueta!(r::GlmResp) = mueta!(r.l, r.mueta, r.eta)

function updatemu!{T<:FP}(r::GlmResp{T}, linPr::Vector{T})
    n = length(linPr)
    length(r.offset) == n ? map!(Add(), r.eta, linPr, r.offset) : copy!(r.eta, linPr)
    linkinv!(r); mueta!(r); var!(r); wrkresid!(r); devresid!(r)
    sum(r.devresid)
end

updatemu!{T<:FP}(r::GlmResp{T}, linPr) = updatemu!(r, convert(Vector{T},vec(linPr)))

var!(r::GlmResp) = var!(r.d, r.var, r.mu)

wrkresid!(r::GlmResp) = map1!(Divide(), map!(Subtract(), r.wrkresid, r.y, r.mu), r.mueta)

function wrkresp(r::GlmResp)
    if length(r.offset) > 0 return map1!(Add(), map(Subtract(), r.eta, r.offset), r.wrkresid) end
    map(Add(), r.eta, r.wrkresid)
end

function wrkwt(r::GlmResp)
    length(r.wts) == 0 && return [r.mueta[i] * r.mueta[i]/r.var[i] for i in 1:length(r.var)]
    [r.wts[i] * r.mueta[i] * r.mueta[i]/r.var[i] for i in 1:length(r.var)]
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
    DataFrame({cc, se, zz, 2.0 * ccdf(Normal(), abs(zz))},
              ["Estimate","Std.Error","z value", "Pr(>|z|)"])
end

function confint(obj::GlmMod, level::Real)
    cft = coeftable(obj)
    hcat(coef(obj),coef(obj)) + cft["Std.Error"] *
    quantile(Normal(), (1. - level)/2.) * [1. -1.]
end
confint(obj::GlmMod) = confint(obj, 0.95)
        
deviance(m::GlmMod)  = deviance(m.rr)

function fit(m::GlmMod; verbose::Bool=false, maxIter::Integer=30, minStepFac::Real=0.001, convTol::Real=1.e-6)
    m.fit && return m
    maxIter >= 1 || error("maxIter must be positive")
    0 < minStepFac < 1 || error("minStepFac must be in (0, 1)")

    cvg = false; p = m.pp; r = m.rr
    scratch = similar(p.X.m)
    devold = updatemu!(r, linpred(delbeta!(p, wrkresp(r), wrkwt(r), scratch)))
    installbeta!(p)
    for i=1:maxIter
        f = 1.0
        dev = updatemu!(r, linpred(delbeta!(p, r.wrkresid, wrkwt(r), scratch)))
        while dev > devold
            f /= 2.; f > minStepFac || error("step-halving failed at beta0 = $beta0")
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
    w = lw == 0 ? ones(T,n) : (T <: Float64 ? copy(wts) : convert(Vector{T}, wts))
    mu = mustart(d, y, w)
    off = T <: Float64 ? copy(offset) : convert(Vector{T}, offset)
    rr = GlmResp{T}(y, d, l, linkfun!(l, similar(mu), mu), mu, off, w)
    res = GlmMod(mf, rr, DensePredChol(mm), f, false)
    dofit ? fit(res) : res
end

glm(e::Expr, df::AbstractDataFrame, d::UnivariateDistribution, l::Link) = glm(Formula(e),df,d,l)
glm(e::Expr, df::AbstractDataFrame, d::UnivariateDistribution) = glm(Formula(e),df,d,canonicallink(d))
glm(f::Formula, df::AbstractDataFrame, d::UnivariateDistribution) = glm(f, df, d, canonicallink(d))
glm(s::String, df::AbstractDataFrame, d::UnivariateDistribution) = glm(Formula(parse(s)[1]), df, d)

## scale(m) -> estimate, s, of the scale parameter
## scale(m,true) -> estimate, s^2, of the squared scale parameter
scale(x::GlmMod,sqr::Bool=false) = 1. # generalize this - only appropriate for Bernoulli and Poisson
