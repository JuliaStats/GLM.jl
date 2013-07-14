type GlmResp{T<:FloatingPoint} <: ModResp               # response in a glm model
    y::Vector{T}                # response
    d::Distribution
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
        insupport(d, y) || error("some elements of y are not in the support of d")
        n = length(y)
        length(eta) == length(mu) == length(wts) == n || error("mismatched sizes")
        lo = length(off); lo == 0 || lo == n || error("offset must have length $n or length 0")
        res = new(y,d,l,Array(T,n),eta,mu,Array(T,n),off,Array(T,n),wts,Array(T,n))
        updatemu!(res, eta)
        res
    end
end

function GlmResp{T<:FloatingPoint}(y::Vector{T}, d::Distribution, l::Link)
    n  = length(y); wt = ones(T,n); mu = mustart(d, y, wt)
    GlmResp{T}(y, d, l, linkfun!(l,Array(T,n),mu), mu, T[], wt)
end
GlmResp{T<:FloatingPoint}(y::Vector{T}, d::Distribution) = GlmResp(y, d, canonicallink(d))
GlmResp{T<:Integer}(y::Vector{T}, d::Distribution, args...) = GlmResp(float64(y), d, args...)

deviance(r::GlmResp) = deviance(r.d, r.mu, r.y, r.wts)

devresid!(r::GlmResp) = devresid!(r.d, r.devresid, r.y, r.mu, r.wts)

linkfun!(r::GlmResp) = linkfun!(r.l, r.eta, r.mu)

linkinv!(r::GlmResp) = linkinv!(r.l, r.mu, r.eta)

mueta!(r::GlmResp) = mueta!(r.l, r.mueta, r.eta)

function updatemu!{T<:FloatingPoint}(r::GlmResp{T}, linPr::Vector{T})
    n = length(linPr)
    length(r.offset) == n ? map!(Add(), r.eta, linPr, r.offset) : copy!(r.eta, linPr)
    linkinv!(r.l, r.mu, r.eta); mueta!(r); var!(r); wrkresid!(r); devresid!(r)
    sum(r.devresid)
end

updatemu!{T<:FloatingPoint}(r::GlmResp{T}, linPr) = updatemu!(r, convert(Vector{T},vec(linPr)))

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
    if mm.fr.terms.intercept
        vnames = UTF8String["(Intercept)"]
    else
        vnames = UTF8String[]
    end
    # Need to only include active levels
    for term in mm.fr.terms.terms
        if isa(mm.fr.df[term], PooledDataArray)
            for lev in levels(mm.fr.df[term])[2:end]
                push!(vnames, string(term, " - ", lev))
            end
        else
            push!(vnames, string(term))
        end
    end
    cc = coef(mm)
    se = stderr(mm)
    zz = cc ./ se
    DataFrame({vnames, cc, se, zz, 2.0 * ccdf(Normal(), abs(zz))},
              ["Term","Estimate","Std.Error","z value", "Pr(>|z|)"])
end

function confint(obj::GlmMod, level::Real)
    cft = coeftable(obj)
    hcat(coef(obj),coef(obj)) + cft["Std.Error"] *
    quantile(Normal(), (1. - level)/2.) * [1. -1.]
end
confint(obj::LmMod) = confint(obj, 0.95)
        
deviance(m::GlmMod)  = deviance(m.rr)

function fit(m::GlmMod; verbose::Bool=false, maxIter::Integer=30, minStepFac::Real=0.001, convTol::Real=1.e-6)
    m.fit && return m
    maxIter >= 1 || error("maxIter must be positive")
    0 < minStepFac < 1 || error("minStepFac must be in (0, 1)")

    cvg = false; p = m.pp; r = m.rr
    scratch = similar(p.X.m)
    devold = updatemu!(r, linpred(delbeta!(p, wrkresp(r), GLM.wrkwt(r), scratch)))
    GLM.installbeta!(p)
    for i=1:maxIter
        f = 1.0
        dev = updatemu!(r, linpred(delbeta!(p, r.wrkresid, GLM.wrkwt(r), scratch)))
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

function glm(f::Formula, df::AbstractDataFrame, d::Distribution, l::Link; dofit::Bool=true)
    mf = ModelFrame(f, df)
    mm = ModelMatrix(mf)
    rr = GlmResp(model_response(mf), d, l)
    res = GlmMod(mf, rr, DensePredChol(mm), f, false)
    dofit ? fit(res) : res
end
glm(f::Formula, df::AbstractDataFrame, d::Distribution) = glm(f, df, d, canonicallink(d))
glm(f::Expr, df::AbstractDataFrame, d::Distribution, l::Link) = glm(Formula(f), df, d, l)
glm(f::Expr, df::AbstractDataFrame, d::Distribution) = glm(Formula(f), df, d, canonicallink(d))
glm(f::String, df::AbstractDataFrame, d::Distribution) = glm(Formula(parse(f)[1]), df, d)

## scale(m) -> estimate, s, of the scale parameter
## scale(m,true) -> estimate, s^2, of the squared scale parameter
scale(x::GlmMod,sqr::Bool=false) = 1. # generalize this - only appropriate for Bernoulli and Poisson
