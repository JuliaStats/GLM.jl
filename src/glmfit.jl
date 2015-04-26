type GlmResp{V<:FPVector,D<:UnivariateDistribution,L<:Link} <: ModResp       # response in a glm model
    y::V                                       # response
    d::D
    l::L
    devresid::V                                # (squared) deviance residuals
    eta::V                                     # linear predictor
    mu::V                                      # mean response
    mueta::V                                   # derivative of mu w.r.t. eta
    offset::V                                  # offset added to linear predictor (usually 0)
    var::V                                     # (unweighted) variance at current mu
    wts::V                                     # prior weights
    wrkwts::V                                  # working weights
    wrkresid::V                                # working residuals
    function GlmResp(y::V, d::D, l::L,
                     eta::V, mu::V,
                     off::V, wts::V)
        if isa(d, Binomial)
            for yy in y; 0. <= yy <= 1. || error("$yy in y is not in [0,1]"); end
        else
            for yy in y; insupport(d, yy) || error("y must be in the support of d"); end
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
    if length(r.offset) == n
        simdmap!(Add(), r.eta, linPr, r.offset)
    else
        copy!(r.eta, linPr)
    end
    linkinv!(r)
    mueta!(r)
    var!(r)
    wrkresid!(r)
    devresid!(r)
    r
end

updatemu!{T<:FPVector}(r::GlmResp{T}, linPr) = updatemu!(r, convert(T,vec(linPr)))

var!{V<:FPVector,L<:Link}(r::GlmResp{V,Binomial,L}) = var!(r.d, r.l, r.var, r.eta)
var!(r::GlmResp) = var!(r.d, r.var, r.mu)

function wrkresid!(r::GlmResp)
    wrkresid = r.wrkresid
    y = r.y
    mu = r.mu
    mueta = r.mueta
    @inbounds @simd for i = 1:length(wrkresid)
        wrkresid[i] = (y[i] - mu[i])/mueta[i]
    end
    wrkresid
end

function wrkresp(r::GlmResp)
    if length(r.offset) > 0
        tmp = r.eta - r.offset
        broadcast!(+, tmp, tmp, r.wrkresid)
    else
        r.eta + r.wrkresid
    end
end

function wrkwt!(r::GlmResp)
    wrkwts = r.wrkwts
    mueta = r.mueta
    var = r.var
    if length(r.wts) == 0
        @simd for i = 1:length(r.var)
            @inbounds wrkwts[i] = abs2(mueta[i])/var[i]
        end
    else
        wts = r.wts
        @simd for i = 1:length(r.var)
            @inbounds wrkwts[i] = wts[i] * abs2(mueta[i])/var[i]
        end
    end
    wrkwts
end

type GeneralizedLinearModel{G<:GlmResp,L<:LinPred} <: LinPredModel
    rr::G
    pp::L
    fit::Bool
end

function coeftable(mm::GeneralizedLinearModel)
    cc = coef(mm)
    se = stderr(mm)
    zz = cc ./ se
    CoefTable(hcat(cc,se,zz,2.0 * ccdf(Normal(), abs(zz))),
              ["Estimate","Std.Error","z value", "Pr(>|z|)"],
              ["x$i" for i = 1:size(mm.pp.X, 2)], 4)
end

function confint(obj::GeneralizedLinearModel, level::Real)
    hcat(coef(obj),coef(obj)) + stderr(obj)*quantile(Normal(),(1. -level)/2.)*[1. -1.]
end
confint(obj::GeneralizedLinearModel) = confint(obj, 0.95)

deviance(m::GeneralizedLinearModel)  = deviance(m.rr)

function _fit(m::GeneralizedLinearModel, verbose::Bool, maxIter::Integer, minStepFac::Real,
              convTol::Real, start)
    m.fit && return m
    maxIter >= 1 || error("maxIter must be positive")
    0 < minStepFac < 1 || error("minStepFac must be in (0, 1)")

    cvg = false; p = m.pp; r = m.rr
    lp = r.mu
    if start != nothing
        copy!(p.beta0, start)
        fill!(p.delbeta, 0)
        linpred!(lp, p, 0)
        updatemu!(r, lp)
    else
        delbeta!(p, wrkresp(r), wrkwt!(r))
        linpred!(lp, p)
        updatemu!(r, lp)
        installbeta!(p)
    end
    devold = deviance(m)
    for i=1:maxIter
        f = 1.0
        local dev
        try
            delbeta!(p, r.wrkresid, wrkwt!(r))
            linpred!(lp, p)
            updatemu!(r, lp)
            dev = deviance(m)
        catch e
            isa(e, DomainError) ? (dev = Inf) : rethrow(e)
        end
        while dev > devold
            f /= 2.; f > minStepFac || error("step-halving failed at beta0 = $(p.beta0)")
            try
                updatemu!(r, linpred(p, f))
                dev = deviance(m)
            catch e
                isa(e, DomainError) ? (dev = Inf) : rethrow(e)
            end
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

StatsBase.fit(m::GeneralizedLinearModel; verbose::Bool=false, maxIter::Integer=30,
              minStepFac::Real=0.001, convTol::Real=1.e-6, start=nothing) =
    _fit(m, verbose, maxIter, minStepFac, convTol, start)

function StatsBase.fit(m::GeneralizedLinearModel, y; wts=nothing, offset=nothing, dofit::Bool=true,
                       verbose::Bool=false, maxIter::Integer=30, minStepFac::Real=0.001, convTol::Real=1.e-6,
                       start=nothing)
    r = m.rr
    V = typeof(r.y)
    r.y = copy!(r.y, y)
    isa(wts, Nothing) || copy!(r.wts, wts)
    isa(offset, Nothing) || copy!(r.offset, offset)
    mustart!(r.d, r.mu, r.y, r.wts)
    linkfun!(r.l, r.eta, r.mu)
    updatemu!(r, r.eta)
    fill!(m.pp.beta0, zero(eltype(m.pp.beta0)))
    m.fit = false
    if dofit
        _fit(m, verbose, maxIter, minStepFac, convTol, start)
    else
        m
    end
end

function StatsBase.fit{T<:FloatingPoint,V<:FPVector}(::Type{GeneralizedLinearModel},
                                                     X::Matrix{T}, y::V, d::UnivariateDistribution,
                                                     l::Link=canonicallink(d);
                                                     dofit::Bool=true,
                                                     wts::V=fill!(similar(y), one(eltype(y))),
                                                     offset::V=similar(y, 0), fitargs...)
    size(X, 1) == size(y, 1) || throw(DimensionMismatch("number of rows in X and y must match"))
    n = length(y)
    length(wts) == n || throw(DimensionMismatch("length(wts) does not match length(y)"))
    length(offset) == n || length(offset) == 0 || throw(DimensionMismatch("length(offset) does not match length(y)"))
    wts = T <: Float64 ? copy(wts) : convert(typeof(y), wts)
    off = T <: Float64 ? copy(offset) : convert(Vector{T}, offset)
    mu = mustart(d, y, wts)
    eta = linkfun!(l, similar(mu), mu)
    if !isempty(off)
        @inbounds @simd for i = 1:length(eta)
            eta[i] -= off[i]
        end
    end
    rr = GlmResp{typeof(y),typeof(d),typeof(l)}(y, d, l, eta, mu, offset, wts)
    res = GeneralizedLinearModel(rr, DensePredChol(X), false)
    dofit ? fit(res; fitargs...) : res
end

StatsBase.fit(::Type{GeneralizedLinearModel}, X::Matrix, y::AbstractVector, d::UnivariateDistribution,
              l::Link=canonicallink(d); kwargs...) =
    fit(GeneralizedLinearModel, float(X), float(y), d, l; kwargs...)

glm(X, y, args...; kwargs...) = fit(GeneralizedLinearModel, X, y, args...; kwargs...)

## scale(m) -> estimate, s, of the scale parameter
## scale(m,true) -> estimate, s^2, of the squared scale parameter
function scale(m::GeneralizedLinearModel, sqr::Bool=false)
    wrkwts = m.rr.wrkwts
    wrkresid = m.rr.wrkresid

    if isa(m.rr.d, Union(Binomial, Poisson))
        return one(eltype(wrkwts))
    end

    s = zero(eltype(wrkwts))
    @inbounds @simd for i = 1:length(wrkwts)
        s += wrkwts[i]*abs2(wrkresid[i])
    end
    s /= df_residual(m)
    sqr ? s : sqrt(s)
end

## Prediction function for GLMs
function predict(mm::GeneralizedLinearModel, newX::AbstractMatrix; offset::FPVector=Array(eltype(newX),0))
    eta = newX * coef(mm)
    if length(mm.rr.offset) > 0
        length(offset) == size(newX, 1) ||
            throw(ArgumentError("fit with offset, so `offset` kw arg must be an offset of length `size(newX, 1)`"))
        simdmap!(Add(), eta, eta, offset)
    else
        length(offset) > 0 && throw(ArgumentError("fit without offset, so value of `offset` kw arg does not make sense"))
    end
    mu = linkinv!(mm.rr.l, eta, eta)
end
