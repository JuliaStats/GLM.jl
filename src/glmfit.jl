"""
    GlmResp

The response vector and various derived vectors in a generalized linear model.
"""
struct GlmResp{V<:FPVector,D<:UnivariateDistribution,L<:Link} <: ModResp
    "`y`: response vector"
    y::V
    d::D
    "`devresid`: the squared deviance residuals"
    devresid::V
    "`eta`: the linear predictor"
    eta::V
    "`mu`: mean response"
    mu::V
    "`offset:` offset added to `Xβ` to form `eta`.  Can be of length 0"
    offset::V
    "`wts:` prior case weights.  Can be of length 0."
    wts::V
    "`wrkwt`: working case weights for the Iteratively Reweighted Least Squares (IRLS) algorithm"
    wrkwt::V
    "`wrkresid`: working residuals for IRLS"
    wrkresid::V
end

function GlmResp(y::V, d::D, l::L, η::V, μ::V, off::V, wts::V) where {V<:FPVector, D, L}
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
    if lw != 0 && lw != n
        throw(DimensionMismatch("wts must have length $n or length 0 but was $lw"))
    end
    if lo != 0 && lo != n
        throw(DimensionMismatch("offset must have length $n or length 0 but was $lo"))
    end

    return GlmResp{V,D,L}(y, d, similar(y), η, μ, off, wts, similar(y), similar(y))
end

function GlmResp(y::V, d::D, l::L, off::V, wts::V) where {V<:FPVector,D,L}
    η   = similar(y)
    μ   = similar(y)
    r   = GlmResp(y, d, l, η, μ, off, wts)
    initialeta!(r.eta, d, l, y, wts, off)
    updateμ!(r, r.eta)
    return r
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
    if !isempty(r.wts)
        map!(*, r.devresid, r.devresid, r.wts)
        map!(*, r.wrkwt, r.wrkwt, r.wts)
    end
    r
end

function updateμ!(r::GlmResp{V,D,L}) where {V<:FPVector,D,L}
    y, η, μ, wrkres, wrkwt, dres = r.y, r.eta, r.mu, r.wrkresid, r.wrkwt, r.devresid

    @inbounds for i in eachindex(y, η, μ, wrkres, wrkwt, dres)
        μi, dμdη = inverselink(L(), η[i])
        μ[i] = μi
        yi = y[i]
        wrkres[i] = (yi - μi) / dμdη
        wrkwt[i] = cancancel(r) ? dμdη : abs2(dμdη) / glmvar(r.d, μi)
        dres[i] = devresid(r.d, yi, μi)
    end
end

function updateμ!(r::GlmResp{V,D,L}) where {V<:FPVector,D<:Union{Bernoulli,Binomial},L<:Link01}
    y, η, μ, wrkres, wrkwt, dres = r.y, r.eta, r.mu, r.wrkresid, r.wrkwt, r.devresid

    @inbounds for i in eachindex(y, η, μ, wrkres, wrkwt, dres)
        μi, dμdη, μomμ = inverselink(L(), η[i])
        μ[i] = μi
        yi = y[i]
        wrkres[i] = (yi - μi) / dμdη
        wrkwt[i] = cancancel(r) ? dμdη : abs2(dμdη) / μomμ
        dres[i] = devresid(r.d, yi, μi)
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
              ["Estimate","Std. Error","z value","Pr(>|z|)","Lower $levstr%","Upper $levstr%"],
              ["x$i" for i = 1:size(mm.pp.X, 2)], 4)
end

function confint(obj::AbstractGLM; level::Real=0.95)
    hcat(coef(obj),coef(obj)) + stderror(obj)*quantile(Normal(),(1. -level)/2.)*[1. -1.]
end

deviance(m::AbstractGLM) = deviance(m.rr)

function loglikelihood(m::AbstractGLM)
    r   = m.rr
    wts = r.wts
    y   = r.y
    mu  = r.mu
    d   = r.d
    ll  = zero(eltype(mu))
    if length(wts) == length(y)
        ϕ = deviance(m)/sum(wts)
        @inbounds for i in eachindex(y, mu, wts)
            ll += loglik_obs(d, y[i], mu[i], wts[i], ϕ)
        end
    else
        ϕ = deviance(m)/length(y)
        @inbounds for i in eachindex(y, mu)
            ll += loglik_obs(d, y[i], mu[i], 1, ϕ)
        end
    end
    ll
end

dof(x::GeneralizedLinearModel) = dispersion_parameter(x.rr.d) ? length(coef(x)) + 1 : length(coef(x))

function _fit!(m::AbstractGLM, verbose::Bool, maxiter::Integer, minstepfac::Real,
               tol::Real, start)

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
        ## The tol*dev term is to avoid failure when deviance
        ## is unchanged except for rouding errors.
        while dev > devold + tol*dev
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
        crit = (devold - dev)/dev
        verbose && println("$i: $dev, $crit")
        if crit < tol || dev == 0
            cvg = true
            break
        end
        @assert isfinite(crit)
        devold = dev
    end
    cvg || throw(ConvergenceException(maxiter))
    m.fit = true
    m
end

function StatsBase.fit!(m::AbstractGLM; verbose::Bool=false, maxiter::Integer=30,
                        minstepfac::Real=0.001, tol::Real=1e-6, start=nothing,
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
        Base.depwarn("'convTol' argument is deprecated, use 'tol' instead", :fit!)
        tol = kwargs[:convTol]
    end
    if !issubset(keys(kwargs), (:maxIter, :minStepFac, :convTol))
        throw(ArgumentError("unsupported keyword argument"))
    end

    _fit!(m, verbose, maxiter, minstepfac, tol, start)
end

function StatsBase.fit!(m::AbstractGLM, y; wts=nothing, offset=nothing, dofit::Bool=true,
                        verbose::Bool=false, maxiter::Integer=30, minstepfac::Real=0.001,
                        tol::Real=1e-6, start=nothing, kwargs...)
    if haskey(kwargs, :maxIter)
        Base.depwarn("'maxIter' argument is deprecated, use 'maxiter' instead", :fit!)
        maxiter = kwargs[:maxIter]
    end
    if haskey(kwargs, :minStepFac)
        Base.depwarn("'minStepFac' argument is deprecated, use 'minstepfac' instead", :fit!)
        minstepfac = kwargs[:minStepFac]
    end
    if haskey(kwargs, :convTol)
        Base.depwarn("'convTol' argument is deprecated, use 'tol' instead", :fit!)
        tol = kwargs[:convTol]
    end
    if !issubset(keys(kwargs), (:maxIter, :minStepFac, :convTol))
        throw(ArgumentError("unsupported keyword argument"))
    end

    r = m.rr
    V = typeof(r.y)
    r.y = copy!(r.y, y)
    isa(wts, Nothing) || copy!(r.wts, wts)
    isa(offset, Nothing) || copy!(r.offset, offset)
    initialeta!(r.eta, r.d, r.l, r.y, r.wts, r.offset)
    updateμ!(r, r.eta)
    fill!(m.pp.beta0, 0)
    m.fit = false
    if dofit
        _fit!(m, verbose, maxiter, minstepfac, tol, start)
    else
        m
    end
end

"""
    fit(GeneralizedLinearModel, X, y, d, [l = canonicallink(d)]; <keyword arguments>)

Fit a generalized linear model to data. `X` and `y` can either be a matrix and a
vector, respectively, or a formula and a data frame. `d` must be a
`UnivariateDistribution`, and `l` must be a [`Link`](@ref), if supplied.

# Keyword Arguments
- `dofit::Bool=true`: Determines whether model will be fit
- `wts::Vector=similar(y,0)`: prior case weights. Can be length 0.
- `offset::Vector=similar(y,0)`: offset added to `Xβ` to form `eta`.  Can be of
length 0
- `verbose::Bool=false`: Display convergence information for each iteration
- `maxiter::Integer=30`: Maximum number of iterations allowed to achieve convergence
- `tol::Real=1e-6`: Convergence is achieved when the relative change in
deviance is less than this
- `minstepfac::Real=0.001`: Minimum line step fraction. Must be between 0 and 1.
- `start::AbstractVector=nothing`: Starting values for beta. Should have the
same length as the number of columns in the model matrix.
"""
function fit(::Type{M},
    X::Union{Matrix{T},SparseMatrixCSC{T}},
    y::V,
    d::UnivariateDistribution,
    l::Link = canonicallink(d);
    dofit::Bool = true,
    wts::V      = similar(y, 0),
    offset::V   = similar(y, 0),
    fitargs...) where {M<:AbstractGLM,T<:FP,V<:FPVector}

    # Check that X and y have the same number of observations
    if size(X, 1) != size(y, 1)
        throw(DimensionMismatch("number of rows in X and y must match"))
    end

    rr = GlmResp(y, d, l, offset, wts)
    res = M(rr, cholpred(X), false)
    return dofit ? fit!(res; fitargs...) : res
end

fit(::Type{M},
X::Union{Matrix,SparseMatrixCSC},
y::AbstractVector,
d::UnivariateDistribution,
l::Link=canonicallink(d); kwargs...) where {M<:AbstractGLM} =
    fit(M, float(X), float(y), d, l; kwargs...)

"""
    glm(F, D, args...; kwargs...)

Fit a generalized linear model to data. Alias for `fit(GeneralizedLinearModel, ...)`.
See [`fit`](@ref) for documentation.
"""
glm(F, D, args...; kwargs...) = fit(GeneralizedLinearModel, F, D, args...; kwargs...)

GLM.Link(mm::AbstractGLM) = mm.l
GLM.Link(r::GlmResp{T,D,L}) where {T,D,L} = L()
GLM.Link(r::GlmResp{T,D,L}) where {T,D<:NegativeBinomial,L<:NegativeBinomialLink} = L(r.d.r)
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
        s = sum(i -> wrkwt[i] * abs2(wrkresid[i]), eachindex(wrkwt, wrkresid)) / dof_residual(m)
        sqr ? s : sqrt(s)
    else
        one(eltype(r.mu))
    end
end

"""
    predict(mm::AbstractGLM, newX::AbstractMatrix; offset::FPVector=Vector{eltype(newX)}(0))

Form the predicted response of model `mm` from covariate values `newX` and, optionally,
an offset.
"""
function predict(mm::AbstractGLM, newX::AbstractMatrix;
                 offset::FPVector=eltype(newX)[])
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

# A helper function to choose default values for eta
function initialeta!(eta::AbstractVector,
                    dist::UnivariateDistribution,
                    link::Link,
                    y::AbstractVector,
                    wts::AbstractVector,
                    off::AbstractVector)


    n  = length(y)
    lw = length(wts)
    lo = length(off)

    if lw == n
        @inbounds @simd for i = eachindex(y, eta, wts)
            μ      = mustart(dist, y[i], wts[i])
            eta[i] = linkfun(link, μ)
        end
    elseif lw == 0
        @inbounds @simd for i = eachindex(y, eta)
            μ      = mustart(dist, y[i], 1)
            eta[i] = linkfun(link, μ)
        end
    else
        throw(ArgumentError("length of wts must be either $n or 0 but was $lw"))
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

