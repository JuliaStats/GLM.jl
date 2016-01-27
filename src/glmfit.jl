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

"""
    updatemu!{T<:FPVector}(r::GlmResp{T}, linPr::T)

Update the GLM response object `r` from the linear predictor `linPr`.
"""
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

    if length(offset) == length(eta)
        broadcast!(+, eta, linPr, offset)
    else
        copy!(eta, linPr)
    end

    @inbounds @simd for i = 1:length(eta)
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

"""
    wrkresp(r::GlmResp)

Return the working response for GLM response `r`.
"""
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

deviance(m::AbstractGLM)  = deviance(m.rr)

function _fit!(m::AbstractGLM, verbose::Bool, maxIter::Integer, minStepFac::Real,
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

StatsBase.fit!(m::AbstractGLM; verbose::Bool=false, maxIter::Integer=30,
              minStepFac::Real=0.001, convTol::Real=1.e-6, start=nothing) =
    _fit!(m, verbose, maxIter, minStepFac, convTol, start)

function initialeta!(dist::UnivariateDistribution, link::Link,
                     eta::AbstractVector, y::AbstractVector, wts::AbstractVector,
                     off::AbstractVector)
    length(eta) == length(y) == length(wts) || throw(DimensionMismatch("argument lengths do not match"))
    @inbounds @simd for i = 1:length(y)
        μ = mustart(dist, y[i], wts[i])
        eta[i] = linkfun(link, μ)
    end
    if !isempty(off)
        @inbounds @simd for i = 1:length(eta)
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
    isa(wts, @compat Void) || copy!(r.wts, wts)
    isa(offset, @compat Void) || copy!(r.offset, offset)
    initialeta!(r.d, r.l, r.eta, r.y, r.wts, r.offset)
    updatemu!(r, r.eta)
    fill!(m.pp.beta0, zero(eltype(m.pp.beta0)))
    m.fit = false
    if dofit
        _fit!(m, verbose, maxIter, minStepFac, convTol, start)
    else
        m
    end
end

function fit{M<:AbstractGLM,T<:FP,V<:FPVector}(::Type{M},
                                               X::@compat(Union{Matrix{T},SparseMatrixCSC{T}}), y::V,
                                               d::UnivariateDistribution,
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
    eta = initialeta!(d, l, similar(y), y, wts, off)
    rr = GlmResp{typeof(y),typeof(d),typeof(l)}(y, d, l, eta, similar(y), offset, wts)
    res = M(rr, cholpred(X), false)
    dofit ? fit!(res; fitargs...) : res
end

fit{M<:AbstractGLM}(::Type{M}, X::@compat(Union{Matrix,SparseMatrixCSC}), y::AbstractVector,
                    d::UnivariateDistribution, l::Link=canonicallink(d); kwargs...) =
    fit(M, float(X), float(y), d, l; kwargs...)

"""
    glm(X, y, family, link; wts, offset, dofit)

Fits the generalized linear model, specified by the symbolic description of the
model along with the family of distribution and a link function.

#### Arguments:
* `X::Matrix`: The formula which is the symbolic representation of the model to fit.
Uses column symbols from the DataFrame data, for example, if names(data)
=[:Y,:X1,:X2], then a valid formula is Y~X1+X2.
* `y::AbstractVector`: Response vector.
* `family::UnivariateDistribution`: Specifies the choice of variance, can be `Bernoulli()`, `Binomial()`, `Gamma()`, `Normal()` or `Poisson()`.
* `link::Link=canonicallink(d)`: This function provides the link function for example, LogitLink() is a valid link for the Binomial() family.
* `dofit::Bool=true`: Indicates whether to fit the model or not.
* `wts::V=fill!(similar(y), one(eltype(y)))`: Weights (inverse dispersion).
* `offset::V=similar(y, 0)`: An additional term added to the linear predictor.

#### Examples:
```
julia> data=DataFrame(X=[1,2,3],Y=[2,4,7])
3×2 DataFrames.DataFrame
│ Row │ X │ Y │
┝━━━━━┿━━━┿━━━┥
│ 1   │ 1 │ 2 │
│ 2   │ 2 │ 4 │
│ 3   │ 3 │ 7 │

julia> OLS = glm(Y~X,data,Normal(),IdentityLink())
DataFrameRegressionModel{GeneralizedLinearModel,Float64}:

Coefficients:
              Estimate Std.Error  z value Pr(>|z|)
(Intercept)  -0.666667   0.62361 -1.06904   0.2850
X                  2.5  0.288675  8.66025   <1e-17
```

"""
glm(X, y, args...; kwargs...) = fit(GeneralizedLinearModel, X, y, args...; kwargs...)

## scale(m) -> estimate, s, of the scale parameter
## scale(m,true) -> estimate, s^2, of the squared scale parameter
function scale(m::AbstractGLM, sqr::Bool=false)
    wrkwts = m.rr.wrkwts
    wrkresid = m.rr.wrkresid

    if isa(m.rr.d, @compat Union{Binomial, Poisson})
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
