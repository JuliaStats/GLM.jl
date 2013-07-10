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
drsum(r::GlmResp) = sum(r.devresid)
linkinv!(r::GlmResp) = linkinv!(r.l, r.mu, r.eta)
mueta!(r::GlmResp) = mueta!(r.l, r.mueta, r.eta)
linkfun!(r::GlmResp) = linkfun!(r.l, r.eta, r.mu)
sqrtwrkwt(r::GlmResp) = map1!(Multiply(), map1!(Sqrt(), map(Divide(),r.wts,r.var)), r.mueta)
var!(r::GlmResp) = var!(r.d, r.var, r.mu)
wrkresid!(r::GlmResp) = map1!(Divide(), map!(Subtract(), r.wrkresid, r.y, r.mu), r.mueta)
function wrkresp(r::GlmResp)
    if length(r.offset) > 0 return map1!(Add(), map(Subtract(), r.eta, r.offset), r.wrkresid) end
    map(Add(), r.eta, r.wrkresid)
end

function updatemu!{T<:FloatingPoint}(r::GlmResp{T}, linPr::Vector{T})
    n = length(linPr)
    length(r.offset) == n ? map!(Add(), r.eta, linPr, r.offset) : copy!(r.eta, linPr)
    linkinv!(r.l, r.mu, r.eta); mueta!(r); var!(r); wrkresid!(r)
    deviance(r)
end
updatemu!{T<:FloatingPoint}(r::GlmResp{T}, linPr) = updatemu!(r, convert(Vector{T},vec(linPr)))

type GlmMod <: LinPredModel
    fr::ModelFrame
    mm::ModelMatrix
    rr::GlmResp
    pp::LinPred
    ff::Formula
    fit::Bool
end
scalepar(x::GlmMod) = 1. # generalize this - only appropriate for Bernoulli and Poisson

## Change this to use optional arguments for the form of the predictor?
function glm(f::Formula, df::AbstractDataFrame, d::Distribution, l::Link, m::DataType)
    m <: LinPred || error("Composite type $m does not extend LinPred")
    mf = ModelFrame(f, df)
    mm = ModelMatrix(mf)
    rr = GlmResp(dv(model_response(mf)), d, l)
    dp = m(mm.m)
    GlmMod(mf, mm, rr, dp, f, false)
end

glm(f::Formula, df::AbstractDataFrame, d::Distribution, l::Link) = glm(f, df, d, l, DensePredQR)
glm(f::Formula, df::AbstractDataFrame, d::Distribution) = glm(f, df, d, canonicallink(d))
glm(f::Expr, df::AbstractDataFrame, d::Distribution) = glm(Formula(f), df, d)
glm(f::String, df::AbstractDataFrame, d::Distribution) = glm(Formula(parse(f)[1]), df, d)

function fit(m::GlmMod; verbose::Bool=false, maxIter::Integer=30, minStepFac::Real=0.001, convTol::Real=1.e-6)
    if !m.fit 
        if maxIter < 1 error("maxIter must be positive") end
        if !(0 < minStepFac < 1) error("minStepFac must be in (0, 1)") end

        cvg = false
        devold = Inf
        p = m.pp
        r = m.rr
        for i=1:maxIter
            delbeta!(p, wrkresp(r), sqrtwrkwt(r))
            dev  = updatemu!(r, linpred(p))
            crit = (devold - dev)/dev
            if verbose println("$i: $dev, $crit") end
            if abs(crit) < convTol
                cvg = true
                break
            end
            if (dev >= devold)
                error("code needed to handle the step-factor case")
            end
            devold = dev
        end
        if !cvg error("failure to converge in $maxIter iterations") end
        installbeta(p)
        m.fit = true
    end
    m
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
confint(obj::LmMod) = confint(obj, 0.95)
