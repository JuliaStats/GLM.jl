"""
    GlmResp

The response vector and various derived vectors in a generalized linear model.
"""
struct GlmResp{V<:FPVector,D<:UnivariateDistribution,L<:Link} <: ModResp
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
    "`weights:` prior case weights.  Can be of length 0."
    weights::V
    "`wrkwt`: working case weights for the Iteratively Reweighted Least Squares (IRLS) algorithm"
    wrkwt::V
    "`wrkresid`: working residuals for IRLS"
    wrkresid::V
end

function GlmResp(y::V, d::D, l::L, η::V, μ::V, off::V, weights::V) where {V<:FPVector, D, L}
    n  = length(y)
    nη = length(η)
    nμ = length(μ)
    lw = length(weights)
    lo = length(off)

    # Check y values
    checky(y, d)

    # Lengths of y, η, and η all need to be n
    if !(nη == nμ == n)
        throw(DimensionMismatch("lengths of η, μ, and y ($nη, $nμ, $n) are not equal"))
    end

    # Lengths of weights and off can be either n or 0
    if lw != 0 && lw != n
        throw(DimensionMismatch("weights must have length $n or length 0 but was $lw"))
    end
    if lo != 0 && lo != n
        throw(DimensionMismatch("offset must have length $n or length 0 but was $lo"))
    end

    return GlmResp{V,D,L}(y, d, l, similar(y), η, μ, off, weights, similar(y), similar(y))
end

function GlmResp(y::FPVector, d::Distribution, l::Link, off::FPVector, weights::FPVector)
    # Instead of convert(Vector{Float64}, y) to be more ForwardDiff friendly
    _y   = convert(Vector{float(eltype(y))}, y)
    _off = convert(Vector{float(eltype(off))}, off)
    _weights = convert(Vector{float(eltype(weights))}, weights)
    η    = similar(_y)
    μ    = similar(_y)
    r    = GlmResp(_y, d, l, η, μ, _off, _weights)
    initialeta!(r.eta, d, l, _y, _weights, _off)
    updateμ!(r, r.eta)
    return r
end

function GlmResp(y::AbstractVector{<:Real}, d::D, l::L, off::AbstractVector{<:Real},
                 weights::AbstractVector{<:Real}) where {D, L}
        GlmResp(float(y), d, l, float(off), float(weights))
end

deviance(r::GlmResp) = sum(r.devresid)

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

function updateμ!(r::GlmResp{T}, linPr::T) where T<:FPVector
    isempty(r.offset) ? copyto!(r.eta, linPr) : broadcast!(+, r.eta, linPr, r.offset)
    updateμ!(r)
    if !isempty(r.weights)
        map!(*, r.devresid, r.devresid, r.weights)
        map!(*, r.wrkwt, r.wrkwt, r.weights)
    end
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
    formula::Union{FormulaTerm,Nothing}
    fit::Bool
    maxiter::Int
    minstepfac::Float64
    atol::Float64
    rtol::Float64
end

GeneralizedLinearModel(rr::GlmResp, pp::LinPred,
                       f::Union{FormulaTerm, Nothing}, fit::Bool) =
    GeneralizedLinearModel(rr, pp, f, fit, 0, NaN, NaN, NaN)

function coeftable(mm::AbstractGLM; level::Real=0.95)
    cc = coef(mm)
    se = stderror(mm)
    zz = cc ./ se
    p = 2 * ccdf.(Ref(Normal()), abs.(zz))
    ci = se*quantile(Normal(), (1-level)/2)
    levstr = isinteger(level*100) ? string(Integer(level*100)) : string(level*100)
    cn = coefnames(mm)
    CoefTable(hcat(cc,se,zz,p,cc+ci,cc-ci),
              ["Coef.","Std. Error","z","Pr(>|z|)","Lower $levstr%","Upper $levstr%"],
              cn, 4, 3)
end

function confint(obj::AbstractGLM; level::Real=0.95)
    hcat(coef(obj),coef(obj)) + stderror(obj)*quantile(Normal(),(1. -level)/2.)*[1. -1.]
end

deviance(m::AbstractGLM) = deviance(m.rr)

function nulldeviance(m::GeneralizedLinearModel)
    r       = m.rr
    weights = weights(r.weights)
    y       = r.y
    d       = r.d
    offset  = r.offset
    hasint  = hasintercept(m)
    dev     = zero(eltype(y))
    if isempty(offset) # Faster method
        if !isempty(weights)
            mu = hasint ?
                mean(y, weights) :
                linkinv(r.link, zero(eltype(y))*zero(eltype(weights))/1)
            @inbounds for i in eachindex(y, weights)
                dev += weights[i] * devresid(d, y[i], mu)
            end
        else
            mu = hasint ? mean(y) : linkinv(r.link, zero(eltype(y))/1)
            @inbounds for i in eachindex(y)
                dev += devresid(d, y[i], mu)
            end
        end
    else
        X = fill(1.0, length(y), hasint ? 1 : 0)
        nullm = fit(GeneralizedLinearModel,
                    X, y, d, r.link; weights=weights, offset=offset,
                    dropcollinear=ispivoted(m.pp),
                    method=decomposition_method(m.pp),
                    maxiter=m.maxiter, minstepfac=m.minstepfac,
                    atol=m.atol, rtol=m.rtol)
        dev = deviance(nullm)
    end
    return dev
end

function loglikelihood(m::AbstractGLM)
    r       = m.rr
    weights = r.weights
    y       = r.y
    mu      = r.mu
    d       = r.d
    ll      = zero(eltype(mu))
    if !isempty(weights)
        ϕ = deviance(m)/sum(weights)
        @inbounds for i in eachindex(y, mu, weights)
            ll += loglik_obs(d, y[i], mu[i], weights[i], ϕ)
        end
    else
        ϕ = deviance(m)/length(y)
        @inbounds for i in eachindex(y, mu)
            ll += loglik_obs(d, y[i], mu[i], 1, ϕ)
        end
    end
    ll
end

function nullloglikelihood(m::GeneralizedLinearModel)
    r       = m.rr
    weights = r.weights
    y       = r.y
    d       = r.d
    offset  = r.offset
    hasint  = hasintercept(m)
    ll  = zero(eltype(y))
    if isempty(r.offset) # Faster method
        if !isempty(weights)
            mu = hasint ? mean(y, weights(weights)) : linkinv(r.link, zero(ll)/1)
            ϕ = nulldeviance(m)/sum(weights)
            @inbounds for i in eachindex(y, weights)
                ll += loglik_obs(d, y[i], mu, weights[i], ϕ)
            end
        else
            mu = hasint ? mean(y) : linkinv(r.link, zero(ll)/1)
            ϕ = nulldeviance(m)/length(y)
            @inbounds for i in eachindex(y)
                ll += loglik_obs(d, y[i], mu, 1, ϕ)
            end
        end
    else
        X = fill(1.0, length(y), hasint ? 1 : 0)
        nullm = fit(GeneralizedLinearModel,
                    X, y, d, r.link; weights=weights, offset=offset,
                    dropcollinear=ispivoted(m.pp),
                    method=decomposition_method(m.pp),
                    maxiter=m.maxiter, minstepfac=m.minstepfac,
                    atol=m.atol, rtol=m.rtol)
        ll = loglikelihood(nullm)
    end
    return ll
end

dof(obj::GeneralizedLinearModel) = linpred_rank(obj) + dispersion_parameter(obj.rr.d)

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
    if start == nothing || isempty(start)
        # Compute beta update based on default response value
        # if no starting values have been passed
        delbeta!(p, wrkresp(r), r.wrkwt)
        linpred!(lp, p)
        updateμ!(r, lp)
        p.beta0 .= p.delbeta
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
        p.beta0 .+= p.delbeta .* f

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

    m.maxiter = maxiter
    m.minstepfac = minstepfac
    m.atol = atol
    m.rtol = rtol

    _fit!(m, verbose, maxiter, minstepfac, atol, rtol, start)
end

function StatsBase.fit!(m::AbstractGLM,
                        y;
                        weights=nothing,
                        offset=nothing,
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

    r = m.rr
    V = typeof(r.y)
    r.y = copy!(r.y, y)
    isa(weights, Nothing) || copy!(r.weights, weights)
    isa(offset, Nothing) || copy!(r.offset, offset)
    initialeta!(r.eta, r.d, r.l, r.y, r.weights, r.offset)
    updateμ!(r, r.eta)
    fill!(m.pp.beta0, 0)
    m.fit = false
    m.maxiter = maxiter
    m.minstepfac = minstepfac
    m.atol = atol
    m.rtol = rtol
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
    - `dofit::Bool=true`: Determines whether model will be fit. Only supported with `glm`.
    $COMMON_FIT_KWARGS_DOCS
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
    dropcollinear::Bool = true,
    method::Symbol = :cholesky,
    dofit::Union{Bool, Nothing} = nothing,
    weights::AbstractVector{<:Real} = similar(y, 0),
    offset::AbstractVector{<:Real}  = similar(y, 0),
    fitargs...) where {M<:AbstractGLM}
    if dofit === nothing
        dofit = true
    else
        Base.depwarn("`dofit` argument to `fit` is deprecated", :fit)
    end

    # Check that X and y have the same number of observations
    if size(X, 1) != size(y, 1)
        throw(DimensionMismatch("number of rows in X and y must match"))
    end

    rr = GlmResp(y, d, l, offset, weights)

    if method === :cholesky
        res = M(rr, cholpred(X, dropcollinear), nothing, false)
    elseif method === :qr
        res = M(rr, qrpred(X, dropcollinear), nothing, false)
    else
        throw(ArgumentError("The only supported values for keyword argument `method` are `:cholesky` and `:qr`."))
    end

    return dofit ? fit!(res; fitargs...) : res
end

fit(::Type{M},
    X::AbstractMatrix,
    y::AbstractVector,
    d::UnivariateDistribution,
    l::Link=canonicallink(d); kwargs...) where {M<:AbstractGLM} =
        fit(M, float(X), float(y), d, l; kwargs...)

function fit(::Type{M},
             f::FormulaTerm,
             data,
             d::UnivariateDistribution,
             l::Link=canonicallink(d);
             offset::Union{AbstractVector, Nothing} = nothing,
             weights::Union{AbstractVector, Nothing} = nothing,
             dropcollinear::Bool = true,
             method::Symbol = :cholesky,
             dofit::Union{Bool, Nothing} = nothing,
             contrasts::AbstractDict{Symbol}=Dict{Symbol,Any}(),
             fitargs...) where {M<:AbstractGLM}
    if dofit === nothing
        dofit = true
    else
        Base.depwarn("`dofit` argument to `fit` is deprecated", :fit)
    end

    f, (y, X) = modelframe(f, data, contrasts, M)

    # Check that X and y have the same number of observations
    if size(X, 1) != size(y, 1)
        throw(DimensionMismatch("number of rows in X and y must match"))
    end

    off = offset === nothing ? similar(y, 0) : offset
    weights = weights === nothing ? similar(y, 0) : weights
    rr = GlmResp(y, d, l, off, weights)

    if method === :cholesky
        res = M(rr, cholpred(X, dropcollinear), f, false)
    elseif method === :qr
        res = M(rr, qrpred(X, dropcollinear), f, false)
    else
        throw(ArgumentError("The only supported values for keyword argument `method` are `:cholesky` and `:qr`."))
    end

    return dofit ? fit!(res; fitargs...) : res
end

"""
    glm(formula, data,
        distr::UnivariateDistribution, link::Link = canonicallink(distr); <keyword arguments>)
    glm(X::AbstractMatrix, y::AbstractVector,
        distr::UnivariateDistribution, link::Link = canonicallink(distr); <keyword arguments>)

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

const PREDICT_COMMON =
"""
`newX` must be either a table (in the [Tables.jl](https://tables.juliadata.org/stable/)
definition) containing all columns used in the model formula, or a matrix with one column
for each predictor in the model. In both cases, each row represents an observation for
which a prediction will be returned.

If `interval=:confidence`, also return upper and lower bounds for a given coverage `level`.
By default (`interval_method = :transformation`) the intervals are constructed by applying
the inverse link to intervals for the linear predictor. If `interval_method = :delta`,
the intervals are constructed by the delta method, i.e., by linearization of the predicted
response around the linear predictor. The `:delta` method intervals are symmetric around
the point estimates, but do not respect natural parameter constraints
(e.g., the lower bound for a probability could be negative).
"""

"""
    predict(mm::AbstractGLM, newX;
            offset::FPVector=[],
            interval::Union{Symbol,Nothing}=nothing, level::Real=0.95,
            interval_method::Symbol=:transformation)

Return the predicted response of model `mm` from covariate values `newX` and,
optionally, an `offset`.

$PREDICT_COMMON
"""
function predict(mm::AbstractGLM, newX::AbstractMatrix;
                 offset::FPVector=eltype(newX)[],
                 interval::Union{Symbol,Nothing}=nothing,
                 level::Real=0.95,
                 interval_method=:transformation)
    r = response(mm)
    len = size(newX, 1)
    res = interval === nothing ?
        similar(r, len) :
        (prediction=similar(r, len), lower=similar(r, len), upper=similar(r, len))
    predict!(res, mm, newX,
             offset=offset, interval=interval, level=level,
             interval_method=interval_method)
end

"""
    predict!(res, mm::AbstractGLM, newX::AbstractMatrix;
             offset::FPVector=eltype(newX)[],
             interval::Union{Symbol,Nothing}=nothing, level::Real=0.95,
             interval_method::Symbol=:transformation)

Store in `res` the predicted response of model `mm` from covariate values `newX`
and, optionally, an `offset`. `res` must be a vector with a length equal to the number
of rows in `newX` if `interval=nothing` (the default), and otherwise a `NamedTuple`
of vectors with names `prediction`, `lower` and `upper`.

$PREDICT_COMMON
"""
function predict!(res::Union{AbstractVector,
                             NamedTuple{(:prediction, :lower, :upper),
                                        <: NTuple{3, AbstractVector}}},
                  mm::AbstractGLM, newX::AbstractMatrix;
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

    if interval === nothing
        res isa AbstractVector ||
            throw(ArgumentError("`res` must be a vector when `interval == nothing` or is omitted"))
        length(res) == size(newX, 1) ||
            throw(DimensionMismatch("length of `res` must equal the number of rows in `newX`"))
        res .= linkinv.(Link(mm), eta)
    elseif interval == :confidence
        res isa NamedTuple ||
            throw(ArgumentError("`res` must be a `NamedTuple` when `interval == :confidence`"))
        mu, lower, upper = res
        length(mu) == length(lower) == length(upper) == size(newX, 1) ||
            throw(DimensionMismatch("length of vectors in `res` must equal the number of rows in `newX`"))
        mu .= linkinv.(Link(mm), eta)
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
            lower .= mu .- normalquantile .* stdmu
            upper .= mu .+ normalquantile .* stdmu
        elseif interval_method == :transformation
            # 2. Construct intervals for eta, then apply inverse link
            lower .= linkinv.(Link(mm), eta .- normalquantile .* stdeta)
            upper .= linkinv.(Link(mm), eta .+ normalquantile .* stdeta)
        else
            throw(ArgumentError("interval_method can be only :transformation or :delta"))
        end
    else
        throw(ArgumentError("only :confidence intervals are defined"))
    end
    return res
end

# A helper function to choose default values for eta
function initialeta!(eta::AbstractVector,
                    dist::UnivariateDistribution,
                    link::Link,
                    y::AbstractVector,
                    weights::AbstractVector,
                    off::AbstractVector)


    n  = length(y)
    lw = length(weights)
    lo = length(off)

    if lw == n
        @inbounds @simd for i = eachindex(y, eta, weights)
            μ      = mustart(dist, y[i], weights[i])
            eta[i] = linkfun(link, μ)
        end
    elseif lw == 0
        @inbounds @simd for i = eachindex(y, eta)
            μ      = mustart(dist, y[i], 1)
            eta[i] = linkfun(link, μ)
        end
    else
        throw(ArgumentError("length of weights must be either $n or 0 but was $lw"))
    end

    if lo == n
        @inbounds @simd for i = eachindex(eta, off)
            eta[i] -= off[i]
        end
    elseif lo != 0
        throw(ArgumentError("length of off must be either $n or 0 but was $lo"))
    end

    return eta
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
