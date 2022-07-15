"""
    GlmResp

The response vector and various derived vectors in a generalized linear model.
"""
struct GlmResp{V<:FPVector, D<:UnivariateDistribution,L<:Link,W<:AbstractWeights{<:Real}} <: ModResp
    "`y`: response vector"
    y::V
    d::D
    "`link`: link function with relevant parameters"
    link::L
    "`devresid`: the squared deviance residuals"
    devresid::V
    "`eta`: the linear predictor"
    eta::V
    "`mu`: mean response"
    mu::V
    "`offset:` offset added to `Xβ` to form `eta`.  Can be of length 0"
    offset::V
    "`wts`: prior case weights.  Can be of length 0."
    wts::W
    "`wrkwt`: working case weights for the Iteratively Reweighted Least Squares (IRLS) algorithm"
    wrkwt::V
    "`wrkresid`: working residuals for IRLS"
    wrkresid::V
end

function GlmResp(y::V, d::D, l::L, η::V, μ::V, off::V, wts::W) where {V<:FPVector, D, L, W}
    n  = length(y)
    nη = length(η)
    nμ = length(μ)
    lw = length(wts)
    lo = length(off)

    # Check y values
    checky(y, d)

    # Lengths of y, η, and η all need to be n
    if !(nη == nμ == n)
        throw(DimensionMismatch("lengths of η, μ, and y ($nη, $nμ, $n) are not equal"))
    end

    # Lengths of wts and off can be either n or 0
    if lw != n
        throw(DimensionMismatch("wts must have length $n but was $lw"))
    end
    if lo != 0 && lo != n
        throw(DimensionMismatch("offset must have length $n or length 0 but was $lo"))
    end

    return GlmResp{V,D,L,W}(y, d, l, similar(y), η, μ, off, wts, similar(y), similar(y))
end

function GlmResp(y::FPVector, d::Distribution, l::Link, off::FPVector, wts::AbstractWeights{<:Real})
    # Instead of convert(Vector{Float64}, y) to be more ForwardDiff friendly
    _y   = convert(Vector{float(eltype(y))}, y)
    _off = convert(Vector{float(eltype(off))}, off)
    η    = similar(_y)
    μ    = similar(_y)
    r    = GlmResp(_y, d, l, η, μ, _off, wts)
    initialeta!(r.eta, d, l, _y, wts, _off)
    updateμ!(r, r.eta)
    return r
end

function GlmResp(y::AbstractVector{<:Real}, d::D, l::L, off::AbstractVector{<:Real}, wts::AbstractWeights{<:Real}) where {D, L}
    GlmResp(float(y), d, l, float(off), wts)
end

deviance(r::GlmResp) = sum(r.devresid)
weights(r::GlmResp) = r.wts
"""
    cancancel(r::GlmResp{V,D,L})

Returns `true` if dμ/dη for link `L` is the variance function for distribution `D`

When `L` is the canonical link for `D` the derivative of the inverse link is a multiple
of the variance function for `D`.  If they are the same a numerator and denominator term in
the expression for the working weights will cancel.
"""
cancancel(::GlmResp) = false
cancancel(::GlmResp{V,D,LogitLink}) where {V,D<:Union{Bernoulli,Binomial}} = true
cancancel(::GlmResp{V,D,NegativeBinomialLink}) where {V,D<:NegativeBinomial} = true
cancancel(::GlmResp{V,D,IdentityLink}) where {V,D<:Normal} = true
cancancel(::GlmResp{V,D,LogLink}) where {V,D<:Poisson} = true

"""
    updateμ!{T<:FPVector}(r::GlmResp{T}, linPr::T)

Update the mean, working weights and working residuals, in `r` given a value of
the linear predictor, `linPr`.
"""
function updateμ! end

function updateμ!(r::GlmResp{T,D,L,<:AbstractWeights}, linPr::T) where {T<:FPVector,D,L}
    isempty(r.offset) ? copyto!(r.eta, linPr) : broadcast!(+, r.eta, linPr, r.offset)
    updateμ!(r)
    map!(*, r.devresid, r.devresid, r.wts)
    map!(*, r.wrkwt, r.wrkwt, r.wts)    
    r
end

function updateμ!(r::GlmResp{T,D,L,<:UnitWeights}, linPr::T) where {T<:FPVector,D,L}
    isempty(r.offset) ? copyto!(r.eta, linPr) : broadcast!(+, r.eta, linPr, r.offset)
    updateμ!(r)
    r
end


function updateμ!(r::GlmResp{V,D,L}) where {V<:FPVector,D,L}
    y, η, μ, wrkres, wrkwt, dres = r.y, r.eta, r.mu, r.wrkresid, r.wrkwt, r.devresid

    @inbounds for i in eachindex(y, η, μ, wrkres, wrkwt, dres)
        μi, dμdη = inverselink(r.link, η[i])
        μ[i] = μi
        yi = y[i]
        wrkres[i] = (yi - μi) / dμdη
        wrkwt[i] = cancancel(r) ? dμdη : abs2(dμdη) / glmvar(r.d, μi)
        dres[i] = devresid(r.d, yi, μi)
    end
end

function _weights_residuals(yᵢ, ηᵢ, μᵢ, omμᵢ, dμdηᵢ, l::LogitLink)
    # LogitLink is the canonical link function for Binomial so only wrkresᵢ can
    # possibly fail when dμdη==0 in which case it evaluates to ±1.
    if iszero(dμdηᵢ)
        wrkresᵢ = ifelse(yᵢ == 1, one(μᵢ), -one(μᵢ))
    else
        wrkresᵢ = ifelse(yᵢ == 1, omμᵢ, yᵢ - μᵢ) / dμdηᵢ
    end
    wrkwtᵢ = μᵢ*omμᵢ

    return wrkresᵢ, wrkwtᵢ
end

function _weights_residuals(yᵢ, ηᵢ, μᵢ, omμᵢ, dμdηᵢ, l::ProbitLink)
    # Since μomμ will underflow before dμdη for Probit, we can just check the
    # former to decide when to evaluate with the tail approximation.
    μomμᵢ = μᵢ*omμᵢ
    if iszero(μomμᵢ)
        wrkresᵢ = 1/abs(ηᵢ)
        wrkwtᵢ  = dμdηᵢ
    else
        wrkresᵢ = ifelse(yᵢ == 1, omμᵢ, yᵢ - μᵢ) / dμdηᵢ
        wrkwtᵢ  = abs2(dμdηᵢ)/μomμᵢ
    end

    return wrkresᵢ, wrkwtᵢ
end

function _weights_residuals(yᵢ, ηᵢ, μᵢ, omμᵢ, dμdηᵢ, l::CloglogLink)
    if yᵢ == 1
        wrkresᵢ = exp(-ηᵢ)
    else
        emη = exp(-ηᵢ)
        if iszero(emη)
            # Diverges to -∞
            wrkresᵢ = oftype(emηᵢ, -Inf)
        elseif isinf(emη)
            # converges to -1
            wrkresᵢ = -one(emη)
        else
            wrkresᵢ = (yᵢ - μᵢ)/omμᵢ*emη
        end
    end

    wrkwtᵢ  = exp(2*ηᵢ)/expm1(exp(ηᵢ))
    # We know that both limits are zero so we'll convert NaNs
    wrkwtᵢ = ifelse(isnan(wrkwtᵢ), zero(wrkwtᵢ), wrkwtᵢ)

    return wrkresᵢ, wrkwtᵢ
end

# Fallback for remaining link functions
function _weights_residuals(yᵢ, ηᵢ, μᵢ, omμᵢ, dμdηᵢ, l::Link01)
    wrkresᵢ = ifelse(yᵢ == 1, omμᵢ, yᵢ - μᵢ)/dμdηᵢ
    wrkwtᵢ  = abs2(dμdηᵢ)/(μᵢ*omμᵢ)

    return wrkresᵢ, wrkwtᵢ
end

function updateμ!(r::GlmResp{V,D,L}) where {V<:FPVector,D<:Union{Bernoulli,Binomial},L<:Link01}
    y, η, μ, wrkres, wrkwt, dres = r.y, r.eta, r.mu, r.wrkresid, r.wrkwt, r.devresid

    @inbounds for i in eachindex(y, η, μ, wrkres, wrkwt, dres)
        yᵢ, ηᵢ = y[i], η[i]
        μᵢ, omμᵢ, dμdηᵢ = inverselink(L(), ηᵢ)
        μ[i] = μᵢ
        # For large values of ηᵢ the quantities dμdη and μomμ will underflow.
        # The ratios defining (yᵢ - μᵢ)/dμdη and dμdη^2/μomμ have fairly stable
        # tail behavior so we can switch algorithm to avoid 0/0. The behavior
        # is specific to the link function so _weights_residuals dispatches to
        # robust versions for LogitLink and ProbitLink
        wrkres[i], wrkwt[i] = _weights_residuals(yᵢ, ηᵢ, μᵢ, omμᵢ, dμdηᵢ, L())
        dres[i] = devresid(r.d, yᵢ, μᵢ)
    end
end

function updateμ!(r::GlmResp{V,D,L}) where {V<:FPVector,D<:NegativeBinomial,L<:NegativeBinomialLink}
    y, η, μ, wrkres, wrkwt, dres = r.y, r.eta, r.mu, r.wrkresid, r.wrkwt, r.devresid

    @inbounds for i in eachindex(y, η, μ, wrkres, wrkwt, dres)
        θ = r.d.r # the shape parameter of the negative binomial distribution
        μi, dμdη, μomμ = inverselink(L(θ), η[i])
        μ[i] = μi
        yi = y[i]
        wrkres[i] = (yi - μi) / dμdη
        wrkwt[i] = dμdη
        dres[i] = devresid(r.d, yi, μi)
    end
end

"""
    wrkresp(r::GlmResp)

The working response, `r.eta + r.wrkresid - r.offset`.
"""
wrkresp(r::GlmResp) = wrkresp!(similar(r.eta), r)

"""
    wrkresp!{T<:FPVector}(v::T, r::GlmResp{T})

Overwrite `v` with the working response of `r`
"""
function wrkresp!(v::T, r::GlmResp{T}) where T<:FPVector
    broadcast!(+, v, r.eta, r.wrkresid)
    isempty(r.offset) ? v : broadcast!(-, v, v, r.offset)
end

abstract type AbstractGLM <: LinPredModel end

mutable struct GeneralizedLinearModel{G<:GlmResp,L<:LinPred} <: AbstractGLM
    rr::G
    pp::L
    fit::Bool
end

function coeftable(mm::AbstractGLM; level::Real=0.95)
    cc = coef(mm)
    se = stderror(mm)
    zz = cc ./ se
    p = 2 * ccdf.(Ref(Normal()), abs.(zz))
    ci = se*quantile(Normal(), (1-level)/2)
    levstr = isinteger(level*100) ? string(Integer(level*100)) : string(level*100)
    CoefTable(hcat(cc,se,zz,p,cc+ci,cc-ci),
              ["Coef.","Std. Error","z","Pr(>|z|)","Lower $levstr%","Upper $levstr%"],
              ["x$i" for i = 1:size(mm.pp.X, 2)], 4, 3)
end

function confint(obj::AbstractGLM; level::Real=0.95)
    hcat(coef(obj),coef(obj)) + stderror(obj)*quantile(Normal(),(1. -level)/2.)*[1. -1.]
end

deviance(m::AbstractGLM) = deviance(m.rr)

loglikelihood(m::AbstractGLM) = loglikelihood(m.rr)

function loglikelihood(r::GlmResp{T,D,L,<:UnitWeights}) where {T,D,L}
    y   = r.y
    mu  = r.mu
    d   = r.d
    ll  = zero(eltype(mu))
    ϕ = deviance(r)/nobs(r)
    @inbounds for i in eachindex(y, mu)
        ll += loglik_obs(d, y[i], mu[i], 1, ϕ)
    end
    return ll
end

function loglikelihood(r::GlmResp{T,D,L,<:FrequencyWeights}) where {T,D,L}
    wts = r.wts
    y   = r.y
    mu  = r.mu
    d   = r.d
    ll  = zero(eltype(mu))
    ϕ = deviance(r)/nobs(r)
    @inbounds for i in eachindex(y, mu, wts)
        ll += loglik_obs(d, y[i], mu[i], wts[i], ϕ)
    end
    return ll
end

function loglikelihood(r::GlmResp{T,D,L,<:AbstractWeights}) where {T,D,L}
    wts = r.wts
    sumwt = sum(wts)
    y   = r.y
    mu  = r.mu
    d   = r.d
    ll  = zero(eltype(mu))
    ϕ = deviance(r)
    n = length(y)    
    @inbounds for i in eachindex(y, mu, wts)
        ll += loglik_aweights_obs(d, y[i], mu[i], wts[i], ϕ, sumwt, n)
    end
    return ll 
end

dof(x::GeneralizedLinearModel) = dispersion_parameter(x.rr.d) ? length(coef(x)) + 1 : length(coef(x))

function _fit!(m::AbstractGLM, verbose::Bool, maxiter::Integer, minstepfac::Real,
               atol::Real, rtol::Real, start)

    # Return early if model has the fit flag set
    m.fit && return m

    # Check arguments
    maxiter >= 1       || throw(ArgumentError("maxiter must be positive"))
    0 < minstepfac < 1 || throw(ArgumentError("minstepfac must be in (0, 1)"))

    # Extract fields and set convergence flag
    cvg, p, r = false, m.pp, m.rr
    lp = r.mu

    # Initialize β, μ, and compute deviance
    if start === nothing || isempty(start)
        # Compute beta update based on default response value
        # if no starting values have been passed
        delbeta!(p, wrkresp(r), r.wrkwt)
        linpred!(lp, p)
        updateμ!(r, lp)
        installbeta!(p)
    else
        # otherwise copy starting values for β
        copy!(p.beta0, start)
        fill!(p.delbeta, 0)
        linpred!(lp, p, 0)
        updateμ!(r, lp)
    end
    devold = deviance(m)

    for i = 1:maxiter
        f = 1.0 # line search factor
        local dev

        # Compute the change to β, update μ and compute deviance
        try
            delbeta!(p, r.wrkresid, r.wrkwt)
            linpred!(lp, p)
            updateμ!(r, lp)
            dev = deviance(m)
        catch e
            isa(e, DomainError) ? (dev = Inf) : rethrow(e)
        end

        # Line search
        ## If the deviance isn't declining then half the step size
        ## The rtol*dev term is to avoid failure when deviance
        ## is unchanged except for rouding errors.
        while dev > devold + rtol*dev
            f /= 2
            f > minstepfac || error("step-halving failed at beta0 = $(p.beta0)")
            try
                updateμ!(r, linpred(p, f))
                dev = deviance(m)
            catch e
                isa(e, DomainError) ? (dev = Inf) : rethrow(e)
            end
        end
        installbeta!(p, f)

        # Test for convergence
        verbose && println("Iteration: $i, deviance: $dev, diff.dev.:$(devold - dev)")
        if devold - dev < max(rtol*devold, atol)
            cvg = true
            break
        end
        @assert isfinite(dev)
        devold = dev
    end
    cvg || throw(ConvergenceException(maxiter))
    m.fit = true
    m
end

function StatsBase.fit!(m::AbstractGLM;
                        verbose::Bool=false,
                        maxiter::Integer=30,
                        minstepfac::Real=0.001,
                        atol::Real=1e-6,
                        rtol::Real=1e-6,
                        start=nothing,
                        kwargs...)
    if haskey(kwargs, :maxIter)
        Base.depwarn("'maxIter' argument is deprecated, use 'maxiter' instead", :fit!)
        maxiter = kwargs[:maxIter]
    end
    if haskey(kwargs, :minStepFac)
        Base.depwarn("'minStepFac' argument is deprecated, use 'minstepfac' instead", :fit!)
        minstepfac = kwargs[:minStepFac]
    end
    if haskey(kwargs, :convTol)
        Base.depwarn("'convTol' argument is deprecated, use `atol` and `rtol` instead", :fit!)
        rtol = kwargs[:convTol]
    end
    if !issubset(keys(kwargs), (:maxIter, :minStepFac, :convTol))
        throw(ArgumentError("unsupported keyword argument"))
    end
    if haskey(kwargs, :tol)
        Base.depwarn("`tol` argument is deprecated, use `atol` and `rtol` instead", :fit!)
        rtol = kwargs[:tol]
    end

    _fit!(m, verbose, maxiter, minstepfac, atol, rtol, start)
end

function StatsBase.fit!(m::AbstractGLM,
                        y;
                        wts=nothing,
                        offset=nothing,
                        dofit::Bool=true,
                        verbose::Bool=false,
                        maxiter::Integer=30,
                        minstepfac::Real=0.001,
                        atol::Real=1e-6,
                        rtol::Real=1e-6,
                        start=nothing,
                        kwargs...)

    if haskey(kwargs, :maxIter)
        Base.depwarn("'maxIter' argument is deprecated, use 'maxiter' instead", :fit!)
        maxiter = kwargs[:maxIter]
    end
    if haskey(kwargs, :minStepFac)
        Base.depwarn("'minStepFac' argument is deprecated, use 'minstepfac' instead", :fit!)
        minstepfac = kwargs[:minStepFac]
    end
    if haskey(kwargs, :convTol)
        Base.depwarn("'convTol' argument is deprecated, use `atol` and `rtol` instead", :fit!)
        rtol = kwargs[:convTol]
    end
    if !issubset(keys(kwargs), (:maxIter, :minStepFac, :convTol))
        throw(ArgumentError("unsupported keyword argument"))
    end
    if haskey(kwargs, :tol)
        Base.depwarn("`tol` argument is deprecated, use `atol` and `rtol` instead", :fit!)
        rtol = kwargs[:tol]
    end

    isa(offset, Nothing) || copy!(r.offset, offset)
    initialeta!(r.eta, r.d, r.l, r.y, r.wts, r.offset)
    updateμ!(r, r.eta)
    fill!(m.pp.beta0, 0)
    m.fit = false
    if dofit
        _fit!(m, verbose, maxiter, minstepfac, atol, rtol, start)
    else
        m
    end
end

const FIT_GLM_DOC = """
    In the first method, `formula` must be a
    [StatsModels.jl `Formula` object](https://juliastats.org/StatsModels.jl/stable/formula/)
    and `data` a table (in the [Tables.jl](https://tables.juliadata.org/stable/) definition, e.g. a data frame).
    In the second method, `X` must be a matrix holding values of the independent variable(s)
    in columns (including if appropriate the intercept), and `y` must be a vector holding
    values of the dependent variable.
    In both cases, `distr` must specify the distribution, and `link` may specify the link
    function (if omitted, it is taken to be the canonical link for `distr`; see [`Link`](@ref)
    for a list of built-in links).

    # Keyword Arguments
    - `dofit::Bool=true`: Determines whether model will be fit
    - `wts::AbstractWeights=aweights(similar(y,0))`: Weights of observations.
      Allowed weights are `AnalyticalWeights`, `FrequencyWeights`, or `ProbabilityWeights`.
      If a vector is passed (deprecated) it is coerced to FrequencyWeights.
      Can be length 0 to indicate no weighting (default).
    - `offset::Vector=similar(y,0)`: offset added to `Xβ` to form `eta`.  Can be of
      length 0
    - `verbose::Bool=false`: Display convergence information for each iteration
    - `maxiter::Integer=30`: Maximum number of iterations allowed to achieve convergence
    - `atol::Real=1e-6`: Convergence is achieved when the relative change in
      deviance is less than `max(rtol*dev, atol)`.
    - `rtol::Real=1e-6`: Convergence is achieved when the relative change in
      deviance is less than `max(rtol*dev, atol)`.
    - `minstepfac::Real=0.001`: Minimum line step fraction. Must be between 0 and 1.
    - `start::AbstractVector=nothing`: Starting values for beta. Should have the
      same length as the number of columns in the model matrix.
    """

"""
    fit(GeneralizedLinearModel, formula, data,
        distr::UnivariateDistribution, link::Link = canonicallink(d); <keyword arguments>)
    fit(GeneralizedLinearModel, X::AbstractMatrix, y::AbstractVector,
        distr::UnivariateDistribution, link::Link = canonicallink(d); <keyword arguments>)

Fit a generalized linear model to data.

$FIT_GLM_DOC
"""
function fit(::Type{M},
    X::AbstractMatrix{<:FP},
    y::AbstractVector{<:Real},
    d::UnivariateDistribution,
    l::Link = canonicallink(d);
    dofit::Bool = true,
    wts::AbstractVector{<:Real} = uweights(length(y)),
    offset::AbstractVector{<:Real} = similar(y, 0),
    fitargs...) where {M<:AbstractGLM}
    # Check that X and y have the same number of observations
    if size(X, 1) != size(y, 1)
        throw(DimensionMismatch("number of rows in X and y must match"))
    end
    # For backward compatibility accept wts as AbstractArray and coerce them to FrequencyWeights
    _wts = if isa(wts, AbstractWeights)
        wts
    elseif isa(wts, AbstractVector)
        Base.depwarn("Passing weights as vector is deprecated in favor of explicitely using " *
                     "AnalyticalWeights, ProbabilityWeights, or FrequencyWeights. Proceeding " *
                     "by coercing wts to `FrequencyWeights`", :fit)
        fweights(wts)
    else
        throw(ArgumentError("`wts` should be an AbstractVector coercible to AbstractWeights"))
    end
    rr = GlmResp(y, d, l, offset, _wts)
    res = M(rr, cholpred(X, false, _wts), false)
    return dofit ? fit!(res; fitargs...) : res
end

fit(::Type{M},
    X::AbstractMatrix,
    y::AbstractVector,
    d::UnivariateDistribution,
    l::Link=canonicallink(d); kwargs...) where {M<:AbstractGLM} =
        fit(M, float(X), float(y), d, l; kwargs...)

"""
    glm(formula, data,
        distr::UnivariateDistribution, link::Link = canonicallink(d); <keyword arguments>)
    glm(X::AbstractMatrix, y::AbstractVector,
        distr::UnivariateDistribution, link::Link = canonicallink(d); <keyword arguments>)

Fit a generalized linear model to data. Alias for `fit(GeneralizedLinearModel, ...)`.

$FIT_GLM_DOC
"""
glm(X, y, args...; kwargs...) = fit(GeneralizedLinearModel, X, y, args...; kwargs...)

GLM.Link(r::GlmResp) = r.link
GLM.Link(m::GeneralizedLinearModel) = Link(m.rr)

Distributions.Distribution(r::GlmResp{T,D,L}) where {T,D,L} = D
Distributions.Distribution(m::GeneralizedLinearModel) = Distribution(m.rr)

"""
    dispersion(m::AbstractGLM, sqr::Bool=false)

Return the estimated dispersion (or scale) parameter for a model's distribution,
generally written σ for linear models and ϕ for generalized linear models.
It is, by definition, equal to 1 for the Bernoulli, Binomial, and Poisson families.

If `sqr` is `true`, the squared dispersion parameter is returned.
"""
function dispersion(m::AbstractGLM, sqr::Bool=false)
    r = m.rr
    if dispersion_parameter(r.d)
        wrkwt, wrkresid = r.wrkwt, r.wrkresid
        dofr = dof_residual(m)
        s = sum(i -> wrkwt[i] * abs2(wrkresid[i]), eachindex(wrkwt, wrkresid)) / dofr
        dofr > 0 || return oftype(s, Inf)
        sqr ? s : sqrt(s)
    else
        one(eltype(r.mu))
    end
end


"""
    predict(mm::AbstractGLM, newX::AbstractMatrix; offset::FPVector=eltype(newX)[],
            interval::Union{Symbol,Nothing}=nothing, level::Real = 0.95,
            interval_method::Symbol = :transformation)

Return the predicted response of model `mm` from covariate values `newX` and,
optionally, an `offset`.

If `interval=:confidence`, also return upper and lower bounds for a given coverage `level`.
By default (`interval_method = :transformation`) the intervals are constructed by applying
the inverse link to intervals for the linear predictor. If `interval_method = :delta`,
the intervals are constructed by the delta method, i.e., by linearization of the predicted
response around the linear predictor. The `:delta` method intervals are symmetric around
the point estimates, but do not respect natural parameter constraints
(e.g., the lower bound for a probability could be negative).
"""
function predict(mm::AbstractGLM, newX::AbstractMatrix;
                 offset::FPVector=eltype(newX)[],
                 interval::Union{Symbol,Nothing}=nothing,
                 level::Real=0.95,
                 interval_method=:transformation)
    eta = newX * coef(mm)
    if !isempty(mm.rr.offset)
        length(offset) == size(newX, 1) ||
            throw(ArgumentError("fit with offset, so `offset` kw arg must be an offset of length `size(newX, 1)`"))
        broadcast!(+, eta, eta, offset)
    else
        length(offset) > 0 && throw(ArgumentError("fit without offset, so value of `offset` kw arg does not make sense"))
    end
    mu = linkinv.(Link(mm), eta)

    if interval === nothing
        return mu
    elseif interval == :confidence
        normalquantile = quantile(Normal(), (1 + level)/2)
        # Compute confidence intervals in two steps
        # (2nd step varies depending on `interval_method`)
        # 1. Estimate variance for eta based on variance for coefficients
        #    through the diagonal of newX*vcov(mm)*newX'
        vcovXnewT = vcov(mm)*newX'
        stdeta = [sqrt(dot(view(newX, i, :), view(vcovXnewT, :, i))) for i in axes(newX,1)]

        if interval_method == :delta
            # 2. Now compute the variance for mu based on variance of eta and
            # construct intervals based on that (Delta method)
            stdmu = stdeta .* abs.(mueta.(Link(mm), eta))
            lower = mu .- normalquantile .* stdmu
            upper = mu .+ normalquantile .* stdmu
        elseif interval_method == :transformation
            # 2. Construct intervals for eta, then apply inverse link
            lower = linkinv.(Link(mm), eta .- normalquantile .* stdeta)
            upper = linkinv.(Link(mm), eta .+ normalquantile .* stdeta)
        else
            throw(ArgumentError("interval_method can be only :transformation or :delta"))
        end
    else
        throw(ArgumentError("only :confidence intervals are defined"))
    end
    (prediction = mu, lower = lower, upper = upper)
end

# A helper function to choose default values for eta
function initialeta!(eta::AbstractVector,
                    dist::UnivariateDistribution,
                    link::Link,
                    y::AbstractVector,
                    wts::AbstractWeights,
                    off::AbstractVector)


    n  = length(y)
    lo = length(off)

    _initialeta!(eta, dist, link, y, wts)

    if lo == n
        @inbounds @simd for i = eachindex(eta, off)
            eta[i] -= off[i]
        end
    elseif lo != 0
        throw(ArgumentError("length of off must be either $n or 0 but was $lo"))
    end

    return eta
end

function _initialeta!(eta, dist, link, y, wts::UnitWeights)
    @inbounds @simd for i in eachindex(y, eta)
        μ      = mustart(dist, y[i], 1)
        eta[i] = linkfun(link, μ)
    end
end

function _initialeta!(eta, dist, link, y, wts::AbstractWeights)
    @inbounds @simd for i in eachindex(y, eta)
        μ      = mustart(dist, y[i], wts[i])
        eta[i] = linkfun(link, μ)
    end
end


# Helper function to check that the values of y are in the allowed domain
function checky(y, d::Distribution)
    if any(x -> !insupport(d, x), y)
        throw(ArgumentError("y must be in the support of D"))
    end
    return nothing
end
function checky(y, d::Binomial)
    for yy in y
        0 ≤ yy ≤ 1 || throw(ArgumentError("$yy in y is not in [0,1]"))
    end
    return nothing
end

"""
    nobs(obj::LinearModel)
    nobs(obj::GLM)

For linear and generalized linear models, returns the number of rows, or,
when prior weights of type FrequencyWeights are specified, the sum of weights.
"""
nobs(obj::LinPredModel) = nobs(obj.rr)

nobs(r::LmResp{V,W}) where {V,W} = oftype(sum(one(eltype(r.wts))), length(r.y))
nobs(r::LmResp{V,W}) where {V,W<:FrequencyWeights} = r.wts.sum

nobs(r::GlmResp{V,D,L,W}) where {V,D,L,W<:FrequencyWeights} = r.wts.sum
nobs(r::GlmResp{V,D,L,W}) where {V,D,L,W} = oftype(sum(one(eltype(r.wts))), length(r.y))
