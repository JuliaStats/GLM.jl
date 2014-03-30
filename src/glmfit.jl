type GlmResp{V<:FPStoredVector} <: ModResp   # response in a glm model
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

function updatemu!{T<:FPStoredVector}(r::GlmResp{T}, linPr::T)
    n = length(linPr)
    length(r.offset) == n ? map!(Add(), r.eta, linPr, r.offset) : copy!(r.eta, linPr)
    linkinv!(r); mueta!(r); var!(r); wrkresid!(r); devresid!(r)
    sum(r.devresid)
end

updatemu!{T<:FPStoredVector}(r::GlmResp{T}, linPr) = updatemu!(r, convert(T,vec(linPr)))

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
    rr::GlmResp
    pp::LinPred
    fit::Bool

    GlmMod(rr::GlmResp, pp::LinPred, fit::Bool) = new(rr, pp, fit)
end

function coeftable(mm::GlmMod)
    cc = coef(mm)
    se = stderr(mm)
    zz = cc ./ se
    CoefTable(hcat(cc,se,zz,2.0 * ccdf(Normal(), abs(zz))),
              ["Estimate","Std.Error","z value", "Pr(>|z|)"],
              ["x$i" for i = 1:size(mm.pp.X, 2)], 4)
end

function confint(obj::GlmMod, level::Real)
    hcat(coef(obj),coef(obj)) + stderr(obj)*quantile(Normal(),(1. -level)/2.)*[1. -1.]
end
confint(obj::GlmMod) = confint(obj, 0.95)
        
deviance(m::GlmMod)  = deviance(m.rr)

function StatsBase.fit(m::GlmMod; verbose::Bool=false, maxIter::Integer=30,
                       minStepFac::Real=0.001, convTol::Real=1.e-6)
    m.fit && return m
    maxIter >= 1 || error("maxIter must be positive")
    0 < minStepFac < 1 || error("minStepFac must be in (0, 1)")

    cvg = false; p = m.pp; r = m.rr
    scratch = similar(p.X)
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

function StatsBase.fit(m::GlmMod, y; wts=nothing, offset=nothing, fitargs...)
    r = m.rr
    V = typeof(r.y)
    r.y = isa(y, V) ? copy(y) : convert(V, y)
    wts == nothing || (r.wts = isa(wts, V) ? copy(wts) : convert(V, wts))
    offset == nothing || (r.offset = isa(offset, V) ? copy(offset) : convert(V, offset))
    r.mu = mustart(r.d, r.y, r.wts)
    linkfun!(r.l, r.eta, r.mu)
    updatemu!(r, r.eta)
    fill!(m.pp.beta0, zero(eltype(m.pp.beta0)))
    m.fit = false
    dofit ? fit(m; fitargs...) : m
end

function StatsBase.fit{T<:FloatingPoint,V<:FPStoredVector}(::Type{GlmMod},
                                                           X::Matrix{T}, y::V, d::UnivariateDistribution,
                                                           l::Link=canonicallink(d);
                                                           dofit::Bool=true,
                                                           wts::V=fill!(similar(y), one(eltype(y))),
                                                           offset::V=similar(y, 0), fitargs...)
    size(X, 1) == size(y, 1) || DimensionMismatch("number of rows in X and y must match")
    n = length(y)
    length(wts) == n || error("length(wts) = $lw should be 0 or $n")
    wts = T <: Float64 ? copy(wts) : convert(typeof(y), wts)
    off = T <: Float64 ? copy(offset) : convert(Vector{T}, offset)
    mu = mustart(d, y, wts)
    eta = linkfun!(l, similar(mu), mu)
    if !isempty(off)
        subtract!(eta, off)
    end
    rr = GlmResp{typeof(y)}(y, d, l, eta, mu, offset, wts)
    res = GlmMod(rr, DensePredChol(X), false)
    dofit ? fit(res; fitargs...) : res
end

StatsBase.fit(::Type{GlmMod}, X::Matrix, y::AbstractVector, args...; kwargs...) =
    fit(GlmMod, float(X), float(y), args...; kwargs...)

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
