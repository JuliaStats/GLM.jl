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
    function GlmResp(y::V, d::D, l::L, eta::V, mu::V, off::V, wts::V)
        if isa(d, Binomial)
            for yy in y
                0. <= yy <= 1. || error("$yy in y is not in [0,1]")
            end
        else
            for yy in y
                insupport(d, yy) || error("y must be in the support of d")
            end
        end
        n = length(y)
        length(eta) == length(mu) == length(wts) == n || error("mismatched sizes")
        lo = length(off)
        lo == 0 || lo == n || error("offset must have length $n or length 0")
        res = new(y,d,l,similar(y),eta,mu,similar(y),off,similar(y),wts,similar(y),similar(y))
        updatemu!(res, eta)
        res
    end
end

# returns the sum of the squared deviance residuals
deviance(r::GlmResp) = sum(r.devresid)

function updatemu!{T<:FPVector}(r::GlmResp{T}, linPr::T)
    y = r.y
    dist = r.d
    link = r.l
    eta = r.eta
    mu = r.mu
    muetav = r.mueta
    offset = r.offset
    var = r.var
    wts = r.wts
    wrkresid = r.wrkresid
    devresidv = r.devresid

    if isempty(offset)
        copy!(eta, linPr)
    else
        broadcast!(+, eta, linPr, offset)
    end

    @inbounds @simd for i = eachindex(eta,mu,muetav,var,y,wrkresid,devresidv)
        η = eta[i]

        # apply the inverse link function generating the mean vector (μ) from the linear predictor (η)
        μ = mu[i] = linkinv(link, η)

        # evaluate the mueta vector (derivative of μ w.r.t. η) from the linear predictor (eta)
        dμdη = muetav[i] = mueta(link, η)

        var[i] = glmvar(dist, link, μ, η)
        ys = y[i]
        wrkresid[i] = (ys - μ)/dμdη
        devresidv[i] = devresid(dist, ys, μ, wts[i])
    end
    r
end

function wrkresp(r::GlmResp)
    tmp = r.eta + r.wrkresid
    isempty(r.offset) ? tmp : broadcast!(-, tmp, tmp, r.offset)
end

function wrkwt!(r::GlmResp)
    wrkwts = r.wrkwts
    mueta = r.mueta
    var = r.var
    if isempty(r.wts)
        @simd for i = eachindex(var,wrkwts,mueta)
            @inbounds wrkwts[i] = abs2(mueta[i])/var[i]
        end
    else
        wts = r.wts
        @simd for i = eachindex(var,wrkwts,wts,mueta,var)
            @inbounds wrkwts[i] = wts[i] * abs2(mueta[i])/var[i]
        end
    end
    wrkwts
end

abstract AbstractGLM <: LinPredModel

type GeneralizedLinearModel{G<:GlmResp,L<:LinPred} <: AbstractGLM
    rr::G
    pp::L
    fit::Bool
end

function coeftable(mm::AbstractGLM)
    cc = coef(mm)
    se = stderr(mm)
    zz = cc ./ se
    CoefTable(hcat(cc,se,zz,2.0 * ccdf(Normal(), abs(zz))),
              ["Estimate","Std.Error","z value", "Pr(>|z|)"],
              ["x$i" for i = 1:size(mm.pp.X, 2)], 4)
end

function confint(obj::AbstractGLM, level::Real)
    hcat(coef(obj),coef(obj)) + stderr(obj)*quantile(Normal(),(1. -level)/2.)*[1. -1.]
end
confint(obj::AbstractGLM) = confint(obj, 0.95)

deviance(m::AbstractGLM) = deviance(m.rr)

function loglikelihood(m::AbstractGLM)
    r = m.rr
    wts = r.wts
    y = r.y
    mu = r.mu
    ϕ = deviance(m)/sum(wts)
    d = r.d
    ll = zero(one(eltype(wts)) * one(loglik_obs(d, y[1], mu[1], wts[1], ϕ)))
    @inbounds for i in eachindex(y, mu, wts)
        ll += loglik_obs(d, y[i], mu[i], wts[i], ϕ)
    end
    ll
end

df(x::GeneralizedLinearModel) = dispersion_parameter(x.rr.d) ? length(coef(x)) + 1 : length(coef(x))

function _fit!(m::AbstractGLM, verbose::Bool, maxIter::Integer, minStepFac::Real,
              convTol::Real, start)
    m.fit && return m
    maxIter >= 1 || error("maxIter must be positive")
    0 < minStepFac < 1 || error("minStepFac must be in (0, 1)")

    cvg, p, r = false, m.pp, m.rr
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
            f /= 2.
            f > minStepFac || error("step-halving failed at beta0 = $(p.beta0)")
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
        if crit < convTol
            cvg = true
            break
        end
        devold = dev
    end
    cvg || error("failure to converge in $maxIter iterations")
    m.fit = true
    m
end

StatsBase.fit!(m::AbstractGLM; verbose::Bool=false, maxIter::Integer=30,
              minStepFac::Real=0.001, convTol::Real=1.e-6, start=nothing) =
    _fit!(m, verbose, maxIter, minStepFac, convTol, start)

function initialeta!(dist::UnivariateDistribution, link::Link,
                     eta::AbstractVector, y::AbstractVector, wts::AbstractVector,
                     off::AbstractVector)
    length(eta) == length(y) == length(wts) || throw(DimensionMismatch("argument lengths do not match"))
    @inbounds @simd for i = eachindex(y,eta,wts)
        μ = mustart(dist, y[i], wts[i])
        eta[i] = linkfun(link, μ)
    end
    if !isempty(off)
        @inbounds @simd for i = eachindex(eta,off)
            eta[i] -= off[i]
        end
    end
    eta
end

function StatsBase.fit!(m::AbstractGLM, y; wts=nothing, offset=nothing, dofit::Bool=true,
                        verbose::Bool=false, maxIter::Integer=30, minStepFac::Real=0.001, convTol::Real=1.e-6,
                        start=nothing)
    r = m.rr
    V = typeof(r.y)
    r.y = copy!(r.y, y)
    isa(wts, Void) || copy!(r.wts, wts)
    isa(offset, Void) || copy!(r.offset, offset)
    initialeta!(r.d, r.l, r.eta, r.y, r.wts, r.offset)
    updatemu!(r, r.eta)
    fill!(m.pp.beta0, 0)
    m.fit = false
    if dofit
        _fit!(m, verbose, maxIter, minStepFac, convTol, start)
    else
        m
    end
end

function fit{M<:AbstractGLM,T<:FP,V<:FPVector}(::Type{M},
    X::Union{Matrix{T},SparseMatrixCSC{T}}, y::V,
    d::UnivariateDistribution,
    l::Link = canonicallink(d);
    dofit::Bool = true,
    wts::V = ones(y),
    offset::V = similar(y, 0), fitargs...)

    size(X, 1) == size(y, 1) || throw(DimensionMismatch("number of rows in X and y must match"))
    n = length(y)
    length(wts) == n || throw(DimensionMismatch("length(wts) does not match length(y)"))
    if length(offset) != n && length(offset) != 0
        throw(DimensionMismatch("length(offset) does not match length(y)"))
    end

    wts = T <: Float64 ? copy(wts) : convert(typeof(y), wts)
    off = T <: Float64 ? copy(offset) : convert(Vector{T}, offset)
    eta = initialeta!(d, l, similar(y), y, wts, off)
    rr = GlmResp{typeof(y), typeof(d), typeof(l)}(y, d, l, eta, similar(y), offset, wts)
    res = M(rr, cholpred(X), false)
    dofit ? fit!(res; fitargs...) : res
end

fit{M<:AbstractGLM}(::Type{M},
    X::Union{Matrix,SparseMatrixCSC},
    y::AbstractVector,
    d::UnivariateDistribution,
    l::Link=canonicallink(d); kwargs...) =
    fit(M, float(X), float(y), d, l; kwargs...)

glm(X, y, args...; kwargs...) = fit(GeneralizedLinearModel, X, y, args...; kwargs...)

"""
    dispersion(m::AbstractGLM, sqr::Bool=false)

    Estimated dispersion (or scale) parameter for a model's distribution,
    generally written σ for linear models and ϕ for generalized linear models.
    It is by definition equal to 1 for Binomial and Poisson families.

    If `sqr` is `true`, the squared parameter is returned.
"""
function dispersion(m::AbstractGLM, sqr::Bool=false)
    wrkwts = m.rr.wrkwts
    wrkresid = m.rr.wrkresid

    if isa(m.rr.d, Union{Binomial, Poisson})
        return one(eltype(wrkwts))
    end

    s = zero(eltype(wrkwts))
    @inbounds @simd for i = eachindex(wrkwts,wrkresid)
        s += wrkwts[i]*abs2(wrkresid[i])
    end
    s /= df_residual(m)
    sqr ? s : sqrt(s)
end

## Prediction function for GLMs
function predict(mm::AbstractGLM, newX::AbstractMatrix; offset::FPVector=Array(eltype(newX),0))
    eta = newX * coef(mm)
    if length(mm.rr.offset) > 0
        length(offset) == size(newX, 1) ||
            throw(ArgumentError("fit with offset, so `offset` kw arg must be an offset of length `size(newX, 1)`"))
        broadcast!(+, eta, eta, offset)
    else
        length(offset) > 0 && throw(ArgumentError("fit without offset, so value of `offset` kw arg does not make sense"))
    end
    mu = [linkinv(mm.rr.l, x) for x in eta]
end
