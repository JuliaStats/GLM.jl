"""
    LmResp

Encapsulates the response for a linear model

# Members

- `mu`: current value of the mean response vector or fitted value
- `offset`: optional offset added to the linear predictor to form `mu`
- `wts`: optional weights for observations (as `AbstractWeights`)
- `y`: observed response vector

Either or both `offset` and `wts` may be of length 0
"""
mutable struct LmResp{V<:FPVector, W<:AbstractWeights} <: ModResp  # response in a linear model
    mu::V                                  # mean response
    offset::V                              # offset added to linear predictor (may have length 0)
    wts::W                                 # prior weights (may have length 0)
    y::V                                   # response
    function LmResp{V, W}(mu::V, off::V, wts::W, y::V) where {V, W}
        n = length(y)
        length(mu) == n || error("mismatched lengths of mu and y")
        ll = length(off)
        ll == 0 || ll == n || error("length of offset is $ll, must be $n or 0")
        ll = length(wts)
        ll == n || error("length of wts is $ll, must be $n")
        new{V,W}(mu, off, wts, y)
    end
end

function LmResp(y::AbstractVector{<:Real}, wts::AbstractWeights)
    # Instead of convert(Vector{Float64}, y) to be more ForwardDiff friendly
    _y = convert(Vector{float(eltype(y))}, y)
    return LmResp{typeof(_y), typeof(wts)}(zero(_y), zero(_y), wts, _y)
end

function updateμ!(r::LmResp{V}, linPr::V) where {V<:FPVector}
    n = length(linPr)
    length(r.y) == n || error("length(linPr) is $n, should be $(length(r.y))")
    length(r.offset) == 0 ? copyto!(r.mu, linPr) : broadcast!(+, r.mu, linPr, r.offset)
    deviance(r)
end

updateμ!(r::LmResp{V}, linPr) where {V<:FPVector} = updateμ!(r, convert(V, vec(linPr)))

function deviance(r::LmResp{T,<:AbstractWeights}) where T
    y = r.y
    mu = r.mu
    wts = r.wts    
    if wts isa UnitWeights
        v = zero(eltype(y)) + zero(eltype(y)) 
        @inbounds @simd for i in eachindex(y,mu,wts)
            v += abs2(y[i] - mu[i])
        end
    else
        v = zero(eltype(y)) + zero(eltype(y)) * zero(eltype(wts))
        @inbounds @simd for i in eachindex(y,mu,wts)
            v += abs2(y[i] - mu[i])*wts[i]
        end
    end
    return v
end

weights(r::LmResp) = r.wts

nobs(r::LmResp{<:Any,W}) where {W<:FrequencyWeights} = sum(r.wts)
nobs(r::LmResp{<:Any,W}) where {W<:AbstractWeights} = oftype(sum(one(eltype(r.wts))), length(r.y))

function loglikelihood(r::LmResp{T,<:Union{UnitWeights, FrequencyWeights}}) where T
    n = nobs(r)
    -n/2 * (log(2π * deviance(r)/n) + 1)
end

function loglikelihood(r::LmResp{T,<:AbstractWeights}) where T
    N = length(r.y)
    n = sum(log, weights(r))
    0.5*(n - N * (log(2π * deviance(r)/N) + 1))
end

function loglikelihood(r::LmResp{T,<:ProbabilityWeights}) where T
    throw(ArgumentError("The `loglikelihood` for probability weighted models is not currently supported."))
end

function residuals(r::LmResp; weighted=false)
    wts = weights(r)
    res = r.y - r.mu
    if !weighted
        res
    elseif r.wts isa AbstractWeights
        sqrt.(wts).*res
    else
        res
    end
end

"""
    LinearModel

A combination of a [`LmResp`](@ref) and a [`LinPred`](@ref)

# Members

- `rr`: a `LmResp` object
- `pp`: a `LinPred` object
"""
struct LinearModel{L<:LmResp,T<:LinPred} <: LinPredModel
    rr::L
    pp::T
end

LinearAlgebra.cholesky(x::LinearModel) = cholesky(x.pp)

function StatsBase.fit!(obj::LinearModel)
    delbeta!(obj.pp, obj.rr.y)
    installbeta!(obj.pp)
    updateμ!(obj.rr, linpred(obj.pp, zero(eltype(obj.rr.y))))
    return obj
end

const FIT_LM_DOC = """
    In the first method, `formula` must be a
    [StatsModels.jl `Formula` object](https://juliastats.org/StatsModels.jl/stable/formula/)
    and `data` a table (in the [Tables.jl](https://tables.juliadata.org/stable/) definition, e.g. a data frame).
    In the second method, `X` must be a matrix holding values of the independent variable(s)
    in columns (including if appropriate the intercept), and `y` must be a vector holding
    values of the dependent variable.

    The keyword argument `wts` can be an `AbstractWeights` vector specifying prior weights
    for observations. Allowed types are:
    - `UnitWeights`: no weighting (all weights equal to 1).
    - `AnalyticaWeights`: describe a non-random relative importance (usually between 0 and 1)
      for each observation.
    - `FrequencyWeights`: describe the number of times (or frequency) each observation was seen.
    - `ProbabilityWeights`: represent the inverse of the sampling probability for each observation,
      providing a correction mechanism for under- or over-sampling certain population groups.
    These weights give equal point estimates but different standard errors.
    If a non-`AbstractWeights` vector is passed (deprecated), it is coerced to `FrequencyWeights`.

    `dropcollinear` controls whether or not `lm` accepts a model matrix which
    is less-than-full rank. If `true` (the default), only the first of each set of
    linearly-dependent columns is used. The coefficient for redundant linearly dependent columns is
    `0.0` and all associated statistics are set to `NaN`.
    """

"""
    fit(LinearModel, formula, data, allowrankdeficient=false;
       [wts::AbstractVector], dropcollinear::Bool=true)
    fit(LinearModel, X::AbstractMatrix, y::AbstractVector;
        wts::AbstractVector=similar(y, 0), dropcollinear::Bool=true)

Fit a linear model to data.

$FIT_LM_DOC
"""
function fit(::Type{LinearModel}, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real},
             allowrankdeficient_dep::Union{Bool,Nothing}=nothing;
             wts::AbstractVector{<:Real}=uweights(length(y)),
             dropcollinear::Bool=true)
    if allowrankdeficient_dep !== nothing
        @warn "Positional argument `allowrankdeficient` is deprecated, use keyword " *
              "argument `dropcollinear` instead. Proceeding with positional argument value: $allowrankdeficient_dep"
        dropcollinear = allowrankdeficient_dep
    end
    # For backward compatibility accept wts as AbstractArray and coerce them to FrequencyWeights
    _wts = if wts isa AbstractWeights
        wts
    elseif wts isa AbstractVector
        Base.depwarn("Passing weights as vector is deprecated in favor of explicitly using " *
                     "`AnalyticalWeights`, `ProbabilityWeights`, or `FrequencyWeights`. Proceeding " *
                     "by coercing `wts` to `FrequencyWeights`", :fit)
        fweights(wts)
    else
        throw(ArgumentError("`wts` should be an `AbstractVector` coercible to `AbstractWeights`"))
    end
    fit!(LinearModel(LmResp(y, _wts), cholpred(X, dropcollinear, _wts)))
end

"""
    lm(formula, data, allowrankdeficient=false;
       [wts::AbstractVector], dropcollinear::Bool=true)
    lm(X::AbstractMatrix, y::AbstractVector;
       wts::AbstractVector=similar(y, 0), dropcollinear::Bool=true)

Fit a linear model to data.
An alias for `fit(LinearModel, X, y; wts=wts, dropcollinear=dropcollinear)`

$FIT_LM_DOC
"""
lm(X, y, allowrankdeficient_dep::Union{Bool,Nothing}=nothing; kwargs...) =
    fit(LinearModel, X, y, allowrankdeficient_dep; kwargs...)

dof(x::LinearModel) = length(coef(x)) + 1

dof(obj::LinearModel{<:LmResp,<:DensePredChol{<:Real,<:CholeskyPivoted}}) = obj.pp.chol.rank + 1

"""
    deviance(obj::LinearModel)

For linear models, the deviance is equal to the residual sum of squares (RSS).
"""
deviance(obj::LinearModel) = deviance(obj.rr)

"""
    nulldeviance(obj::LinearModel)

For linear models, the deviance of the null model is equal to the total sum of squares (TSS).
"""
function nulldeviance(obj::LinearModel)
    y = obj.rr.y
    wts = weights(obj)
    
    if hasintercept(obj)
        m = mean(y, wts)
    else
        @warn("Starting from GLM.jl 1.8, null model is defined as having no predictor at all " *
              "when a model without an intercept is passed.")
        m = zero(eltype(y))
    end

    v = zero(eltype(y))*zero(eltype(wts))
    if wts isa UnitWeights
        @inbounds @simd for i = eachindex(y,wts)
            v += abs2(y[i] - m)
        end
    else
        @inbounds @simd for i = eachindex(y,wts)
            v += abs2(y[i] - m)*wts[i]
        end
    end
    return v
end

function nullloglikelihood(m::LinearModel) 
    wts = weights(m)
    if wts isa Union{UnitWeights, FrequencyWeights}
        n = nobs(m)
        -n/2 * (log(2π * nulldeviance(m)/n) + 1)
    else
        N = length(m.rr.y)
        n = sum(log, wts)
        0.5*(n - N * (log(2π * nulldeviance(m)/N) + 1))
    end
end

loglikelihood(obj::LinearModel) = loglikelihood(obj.rr)


r2(obj::LinearModel) = 1 - deviance(obj)/nulldeviance(obj)
adjr2(obj::LinearModel) = 1 - (1 - r²(obj))*(nobs(obj)-hasintercept(obj))/dof_residual(obj)

function dispersion(x::LinearModel, sqr::Bool=false)
    dofr = dof_residual(x)
    ssqr = deviance(x.rr)/dofr
    dofr > 0 || return oftype(ssqr, Inf)
    return sqr ? ssqr : sqrt(ssqr)
end

function coeftable(mm::LinearModel; level::Real=0.95)
    cc = coef(mm)
    dofr = dof_residual(mm)
    se = stderror(mm)
    tt = cc ./ se
    if dofr > 0
        p = ccdf.(Ref(FDist(1, dofr)), abs2.(tt))
        ci = se*quantile(TDist(dofr), (1-level)/2)
    else
        p = [isnan(t) ? NaN : 1.0 for t in tt]
        ci = [isnan(t) ? NaN : -Inf for t in tt]
    end
    levstr = isinteger(level*100) ? string(Integer(level*100)) : string(level*100)
    CoefTable(hcat(cc,se,tt,p,cc+ci,cc-ci),
              ["Coef.","Std. Error","t","Pr(>|t|)","Lower $levstr%","Upper $levstr%"],
              ["x$i" for i = 1:size(mm.pp.X, 2)], 4, 3)
end

"""
    predict(mm::LinearModel, newx::AbstractMatrix;
            interval::Union{Symbol,Nothing} = nothing, level::Real = 0.95)

If `interval` is `nothing` (the default), return a vector with the predicted values
for model `mm` and new data `newx`.
Otherwise, return a vector with the predicted values, as well as vectors with
the lower and upper confidence bounds for a given `level` (0.95 equates alpha = 0.05).
Valid values of `interval` are `:confidence` delimiting the  uncertainty of the
predicted relationship, and `:prediction` delimiting estimated bounds for new data points.
"""
function predict(mm::LinearModel, newx::AbstractMatrix;
                 interval::Union{Symbol,Nothing}=nothing, level::Real = 0.95)
    retmean = newx * coef(mm)
    if interval === :confint
        Base.depwarn("interval=:confint is deprecated in favor of interval=:confidence", :predict)
        interval = :confidence
    end
    if interval === nothing
        return retmean
    elseif mm.pp.chol isa CholeskyPivoted &&
        mm.pp.chol.rank < size(mm.pp.chol, 2)
        throw(ArgumentError("prediction intervals are currently not implemented " *
                            "when some independent variables have been dropped " *
                            "from the model due to collinearity"))
    end
    !isweighted(mm) || error("prediction with confidence intervals not yet implemented for weighted regression")
    chol = cholesky!(mm.pp)
    # get the R matrix from the QR factorization
    if chol isa CholeskyPivoted
        ip = invperm(chol.p)
        R = chol.U[ip, ip]
    else
        R = chol.U
    end
    residvar = ones(size(newx,2)) * deviance(mm)/dof_residual(mm)
    if interval == :confidence
        retvariance = (newx/R).^2 * residvar
    elseif interval == :prediction
        retvariance = (newx/R).^2 * residvar .+ deviance(mm)/dof_residual(mm)
    else
        error("only :confidence and :prediction intervals are defined")
    end
    retinterval = quantile(TDist(dof_residual(mm)), (1. - level)/2) * sqrt.(retvariance)
    (prediction = retmean, lower = retmean .+ retinterval, upper = retmean .- retinterval)
end

function confint(obj::LinearModel; level::Real=0.95)
    hcat(coef(obj),coef(obj)) + stderror(obj) *
    quantile(TDist(dof_residual(obj)), (1. - level)/2.) * [1. -1.]
end

function momentmatrix(m::LinearModel; weighted=isweighted(m)) 
    X = modelmatrix(m; weighted=false)
    r = residuals(m; weighted=false)
    if weighted && isweighted(m)
        return X .* r .* weights(m)
    else
        return X .* r
    end
end

"""
    cooksdistance(obj::LinearModel)

Compute [Cook's distance](https://en.wikipedia.org/wiki/Cook%27s_distance)
for each observation in linear model `obj`, giving an estimate of the influence
of each data point.
Currently only implemented for linear models without weights.
"""
## To remove when https://github.com/JuliaStats/StatsAPI.jl/pull/16 is merged
function crossmodelmatrix(model::RegressionModel; weighted::Bool=false) 
    x = weighted ? modelmatrix(model; weighted=weighted) : modelmatrix(model)
    return Symmetric(x' * x)
end
    

function StatsBase.cooksdistance(obj::LinearModel)
    wts = weights(obj)
    u = residuals(obj; weighted=isweighted(obj))
    mse = GLM.dispersion(obj,true)
    k = dof(obj)-1
    d_res = dof_residual(obj)
    X = modelmatrix(obj; weighted=isweighted(obj))    
    if k == size(X,2)         
        XtX = crossmodelmatrix(obj; weighted=isweighted(obj))
        hii = diag(X * inv(XtX) * X')
        D = @. u^2 * (hii / (1 - hii)^2) / (k*mse)
    else
        pp = obj.pp
        C = invchol(pp)
        nancols = [all(isnan, col) for col in eachcol(C)]
        nnancols = .!nancols
        Xc = view(X, :, nnancols)
        XtX = (Xc)'*Xc
        hii = diag(Xc * inv(XtX) * Xc')
        D = @. u^2 * (hii / (1 - hii)^2) / (k*mse)
    end
    return D
end
