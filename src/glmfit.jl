"""
    GlmResp

The response vector and various derived vectors in a generalized linear model.
"""
immutable GlmResp{V<:FPVector,D<:UnivariateDistribution,L<:Link} <: ModResp
    "`y`: response vector"
    y::V
    d::D
    "`devresid`: the squared deviance residuals"
    devresid::V
    "`eta`: the linear predictor"
    eta::V
    "`mu`: mean response"
    mu::V
    "`mueta`: the derivative of `mu` w.r.t. `eta`"
    mueta::V
    "`offset:` offset added to `Xβ` to form `eta`.  Can be of length 0"
    offset::V
    "`wts:` prior case weights.  Can be of length 0."
    wts::V
    "`wrkwt`: working case weights for the Iteratively Reweighted Least Squares (IRLS) algorithm"
    wrkwt::V
    "`wrkresid`: working residuals for IRLS"
    wrkresid::V
end

function GlmResp{V<:FPVector, D, L}(y::V, d::D, l::L, η::V, μ::V, off::V, wts::V)
    if d == Binomial()
        for yy in y
            0. <= yy <= 1. || throw(ArgumentError("$yy in y is not in [0,1]"))
        end
    else
        all(x -> insupport(d, x), y) || throw(ArgumentError("y must be in the support of D"))
    end
    n = length(y)
    nη = length(η)
    nμ = length(μ)
    length(wts) == nη == nμ == n || throw(DimensionMismatch(
        "lengths of η, μ, y and wts ($nη, $nμ, $(length(wts)), $n) are not equal"))
    lo = length(off)
    lo == 0 || lo == n || error("offset must have length $n or length 0")
    res = GlmResp{V,D,L}(y, d, similar(y), η, μ, similar(y), off, wts, similar(y), similar(y))
    updateμ!(res, η)
    res
end

deviance(r::GlmResp) = sum(r.devresid)

"""
    wtscale!{T<:FPVector}(devr::T, wkwt::T, wt::T)

Scale the deviance residuals, `devr`, and the working weights, `wkwt`, by `wt`,
when `wt` is nonempty.
"""
function wtscale!{T<:FPVector}(devr::T, wkwt::T, wt::T)
    if !isempty(wt)
        devr .*= wt
        wkwt .*= wt
    end
end

"""
    updateμ!{T<:FPVector}(r::GlmResp{T}, linPr::T)

Update the mean, working weights and working residuals, in `r` given a value of
the linear predictor, `linPr`.
"""
function updateμ!{T<:FPVector,D,L}(r::GlmResp{T,D,L}, linPr::T)
    isempty(r.offset) ? copy!(r.eta, linPr) : broadcast!(+, r.eta, linPr, r.offset)
    updateμ!(r)
    if !isempty(r.wts)
        r.devresid .*= r.wts
        r.wrkwt .*= r.wts
    end
    r
end

function updateμ!{T<:FPVector,D<:Union{Bernoulli,Binomial}}(r::GlmResp{T,D,LogitLink})
    y, η, μ, wrkres, wrkwt, dres = r.y, r.eta, r.mu, r.wrkresid, r.wrkwt, r.devresid

    @inbounds Threads.@threads for i in eachindex(μ)
        ηi = clamp(η[i], -20.0, 20.0)
        ei = exp(-ηi)
        opei = 1 + ei
        μi = μ[i] = inv(opei)
        dμdη = wrkwt[i] = ei / abs2(opei)
        yi = y[i]
        wrkres[i] = (yi - μi) / dμdη
        dres[i] = -2 * (yi == 1 ? log(μi) : yi == 0 ? log1p(-μi) :
            (yi * (log(μi) - log(yi)) + (1 - yi) * (log1p(-μi) - log1p(-yi))))
    end
end

function updateμ!{T<:FPVector,D<:Poisson}(r::GlmResp{T,D,LogLink})
    y, η, μ, wrkres, wrkwt, dres = r.y, r.eta, r.mu, r.wrkresid, r.wrkwt, r.devresid

    @inbounds Threads.@threads for i in eachindex(η)
        ηi = η[i]
        μi = μ[i] = exp(ηi)
        dμdη = wrkwt[i] = μi
        yi = y[i]
        wrkres[i] = (yi - μi) / dμdη
        dres[i] = 2 * (xlogy(yi, yi / μi) - (yi - μi))
    end
end

function updateμ!{T<:FPVector,D<:Normal}(r::GlmResp{T,D,IdentityLink})
    y, η, μ, wrkres, wrkwt, dres = r.y, r.eta, r.mu, r.wrkresid, r.wrkwt, r.devresid

    @inbounds Threads.@threads for i in eachindex(η)
        μi = μ[i] = η[i]
        wrkwt[i] = 1
        yi = y[i]
        wrkresi = wrkres[i] = (yi - μi)
        dres[i] = abs2(wrkresi)
    end
end

function updateμ!{T,D,L}(r::GlmResp{T,D,L})
    y, η, μ, wrkres, wrkwt, dres = r.y, r.eta, r.mu, r.wrkresid, r.wrkwt, r.devresid

    @inbounds @simd for i = eachindex(y, η, μ, wrkres, wrkwt, dres)
        ηi = η[i]
        # apply the inverse link function generating the mean vector (μ) from the linear predictor (η)
        μi = μ[i] = linkinv(L(), ηi)
        # evaluate the mueta vector (derivative of μ w.r.t. η) from the linear predictor (eta)
        dμdη = mueta(L(), ηi)
        yi = y[i]
        wrkres[i] = (yi - μi)/dμdη
        dres[i] = devresid(r.d, yi, μi)
        wrkwt[i] = abs2(dμdη) / max(eps(), glmvar(r.d, L(), μi, ηi))
    end
end

"""
    wrkresp(r::GlmResp)

The working response, `r.eta + r.wrkresid - r.offset`.
"""
function wrkresp(r::GlmResp)
    tmp = r.eta .+ r.wrkresid
    isempty(r.offset) ? tmp : broadcast!(-, tmp, tmp, r.offset)
end

"""
    wrkresp!{T<:FPVector}(v::T, r::GlmResp{T})

Overwrite `v` with the working response of `r`
"""
function wrkresp{T<:FPVector}(v::T, r::GlmResp{T})
    broadcast!(+, v, r.eta, r.wrkresid)
    isempty(r.offset) ? v : broadcast!(-, v, v, r.offset)
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
    CoefTable(hcat(cc,se,zz,2.0 * ccdf(Normal(), abs.(zz))),
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
    ll = zero(loglik_obs(d, y[1], mu[1], wts[1], ϕ))
    @inbounds for i in eachindex(y, mu, wts)
        ll += loglik_obs(d, y[i], mu[i], wts[i], ϕ)
    end
    ll
end

df(x::GeneralizedLinearModel) = dispersion_parameter(x.rr.d) ? length(coef(x)) + 1 : length(coef(x))

function _fit!(m::AbstractGLM, verbose::Bool, maxIter::Integer, minStepFac::Real,
              convTol::Real, start)
    m.fit && return m
    maxIter >= 1 || throw(ArgumentError("maxIter must be positive"))
    0 < minStepFac < 1 || throw(ArgumentError("minStepFac must be in (0, 1)"))

    cvg, p, r = false, m.pp, m.rr
    lp = r.mu
    if start == nothing || isempty(start)
        delbeta!(p, wrkresp(r), r.wrkwt)
        linpred!(lp, p)
        updateμ!(r, lp)
        installbeta!(p)
    else
        copy!(p.beta0, start)
        fill!(p.delbeta, 0)
        linpred!(lp, p, 0)
        updateμ!(r, lp)
    end
    devold = deviance(m)
    for i = 1:maxIter
        f = 1.0
        local dev
        try
            delbeta!(p, r.wrkresid, r.wrkwt)
            linpred!(lp, p)
            updateμ!(r, lp)
            dev = deviance(m)
        catch e
            isa(e, DomainError) ? (dev = Inf) : rethrow(e)
        end
        while dev > devold
            f /= 2.
            f > minStepFac || error("step-halving failed at beta0 = $(p.beta0)")
            try
                updateμ!(r, linpred(p, f))
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
    @inbounds @simd for i = eachindex(y, eta, wts)
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
    updateμ!(r, r.eta)
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
    rr = GlmResp(y, d, l, eta, similar(y), offset, wts)
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

GLM.Link(mm::AbstractGLM) = mm.l
GLM.Link{T,D,L}(r::GlmResp{T,D,L}) = L()
GLM.Link(m::GeneralizedLinearModel) = Link(m.rr)

Distributions.Distribution{T,D,L}(r::GlmResp{T,D,L}) = D
Distributions.Distribution(m::GeneralizedLinearModel) = Distribution(m.rr)

"""
    dispersion(m::AbstractGLM, sqr::Bool=false)

Estimated dispersion (or scale) parameter for a model's distribution,
generally written σ for linear models and ϕ for generalized linear models.
It is, by definition, equal to 1 for the Bernoulli, Binomial, and Poisson families.

If `sqr` is `true`, the squared parameter is returned.
"""
function dispersion(m::AbstractGLM, sqr::Bool=false)
    r = m.rr
    if dispersion_parameter(r.d)
        wrkwt, wrkresid = r.wrkwt, r.wrkresid
        s = sum(i -> wrkwt[i] * abs2(wrkresid[i]), eachindex(wrkwt, wrkresid)) / df_residual(m)
        sqr ? s : sqrt(s)
    else
        one(eltype(r.mu))
    end
end

"""
    predict(mm::AbstractGLM, newX::AbstractMatrix; offset::FPVector=Array(eltype(newX),0))

Form the predicted response of model `mm` from covariate values `newX` and, optionally,
an offset.
"""
function predict(mm::AbstractGLM, newX::AbstractMatrix; offset::FPVector=Array(eltype(newX),0))
    eta = newX * coef(mm)
    if !isempty(mm.rr.offset)
        length(offset) == size(newX, 1) ||
            throw(ArgumentError("fit with offset, so `offset` kw arg must be an offset of length `size(newX, 1)`"))
        broadcast!(+, eta, eta, offset)
    else
        length(offset) > 0 && throw(ArgumentError("fit without offset, so value of `offset` kw arg does not make sense"))
    end
    mu = [linkinv(Link(mm), x) for x in eta]
end
