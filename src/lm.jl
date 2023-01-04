"""
    LmResp

Encapsulates the response for a linear model

# Members

- `mu`: current value of the mean response vector or fitted value
- `offset`: optional offset added to the linear predictor to form `mu`
- `wts`: optional vector of prior frequency (a.k.a. case) weights for observations
- `y`: observed response vector

Either or both `offset` and `wts` may be of length 0
"""
mutable struct LmResp{V<:FPVector} <: ModResp  # response in a linear model
    mu::V                                  # mean response
    offset::V                              # offset added to linear predictor (may have length 0)
    wts::V                                 # prior weights (may have length 0)
    y::V                                   # response
    function LmResp{V}(mu::V, off::V, wts::V, y::V) where V
        n = length(y)
        length(mu) == n || error("mismatched lengths of mu and y")
        ll = length(off)
        ll == 0 || ll == n || error("length of offset is $ll, must be $n or 0")
        ll = length(wts)
        ll == 0 || ll == n || error("length of wts is $ll, must be $n or 0")
        new{V}(mu, off, wts, y)
    end
end

function LmResp(y::AbstractVector{<:Real}, wts::Union{Nothing,AbstractVector{<:Real}}=nothing)
    # Instead of convert(Vector{Float64}, y) to be more ForwardDiff friendly
    _y = convert(Vector{float(eltype(y))}, y)
    _wts = if wts === nothing
        similar(_y, 0)
    else
        convert(Vector{float(eltype(wts))}, wts)
    end
    return LmResp{typeof(_y)}(zero(_y), zero(_y), _wts, _y)
end

function updateμ!(r::LmResp{V}, linPr::V) where V<:FPVector
    n = length(linPr)
    length(r.y) == n || error("length(linPr) is $n, should be $(length(r.y))")
    length(r.offset) == 0 ? copyto!(r.mu, linPr) : broadcast!(+, r.mu, linPr, r.offset)
    deviance(r)
end

updateμ!(r::LmResp{V}, linPr) where {V<:FPVector} = updateμ!(r, convert(V, vec(linPr)))

function deviance(r::LmResp)
    y = r.y
    mu = r.mu
    wts = r.wts
    v = zero(eltype(y)) + zero(eltype(y)) * zero(eltype(wts))
    if isempty(wts)
        @inbounds @simd for i = eachindex(y,mu)
            v += abs2(y[i] - mu[i])
        end
    else
        @inbounds @simd for i = eachindex(y,mu,wts)
            v += abs2(y[i] - mu[i])*wts[i]
        end
    end
    v
end

function loglikelihood(r::LmResp)
    n = isempty(r.wts) ? length(r.y) : sum(r.wts)
    -n/2 * (log(2π * deviance(r)/n) + 1)
end

residuals(r::LmResp) = r.y - r.mu

"""
    LinearModel

A combination of a [`LmResp`](@ref), a [`LinPred`](@ref),
and possibly a `FormulaTerm`

# Members

- `rr`: a `LmResp` object
- `pp`: a `LinPred` object
- `f`: either a `FormulaTerm` object or `nothing`
"""
struct LinearModel{L<:LmResp,T<:LinPred} <: LinPredModel
    rr::L
    pp::T
    formula::Union{FormulaTerm,Nothing}
end

LinearAlgebra.cholesky(x::LinearModel) = cholesky(x.pp)

function StatsBase.fit!(obj::LinearModel)
    if isempty(obj.rr.wts)
        delbeta!(obj.pp, obj.rr.y)
    else 
        delbeta!(obj.pp, obj.rr.y, obj.rr.wts)
    end
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

    # Keyword Arguments
    $COMMON_FIT_KWARGS_DOCS
    """

"""
    fit(LinearModel, formula::FormulaTerm, data;
        [wts::AbstractVector], dropcollinear::Bool=true, method::Symbol=:cholesky,
        contrasts::AbstractDict{Symbol}=Dict{Symbol,Any}())
    fit(LinearModel, X::AbstractMatrix, y::AbstractVector;
        wts::AbstractVector=similar(y, 0), dropcollinear::Bool=true, method::Symbol=:cholesky)

Fit a linear model to data.

$FIT_LM_DOC
"""
function fit(::Type{LinearModel}, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real},
             allowrankdeficient_dep::Union{Bool,Nothing}=nothing;
             wts::AbstractVector{<:Real}=similar(y, 0),
             dropcollinear::Bool=true,
             method::Symbol=:cholesky)
    if allowrankdeficient_dep !== nothing
        @warn "Positional argument `allowrankdeficient` is deprecated, use keyword " *
              "argument `dropcollinear` instead. Proceeding with positional argument value: $allowrankdeficient_dep"
        dropcollinear = allowrankdeficient_dep
    end
    
    if method === :cholesky
        fit!(LinearModel(LmResp(y, wts), cholpred(X, dropcollinear), nothing))
    elseif method === :qr
        fit!(LinearModel(LmResp(y, wts), qrpred(X, dropcollinear), nothing))
    else
        throw(ArgumentError("The only supported values for keyword argument `method` are `:cholesky` and `:qr`."))
    end
end

function fit(::Type{LinearModel}, f::FormulaTerm, data,
             allowrankdeficient_dep::Union{Bool,Nothing}=nothing;
             wts::Union{AbstractVector{<:Real}, Nothing}=nothing,
             dropcollinear::Bool=true,
             method::Symbol=:cholesky,
             contrasts::AbstractDict{Symbol}=Dict{Symbol,Any}())
    f, (y, X) = modelframe(f, data, contrasts, LinearModel)
    wts === nothing && (wts = similar(y, 0))

    if method === :cholesky
        fit!(LinearModel(LmResp(y, wts), cholpred(X, dropcollinear), f))
    elseif method === :qr
        fit!(LinearModel(LmResp(y, wts), qrpred(X, dropcollinear), f))
    else
        throw(ArgumentError("The only supported values for keyword argument `method` are `:cholesky` and `:qr`."))
    end
end

"""
    lm(formula, data;
       [wts::AbstractVector], dropcollinear::Bool=true, method::Symbol=:cholesky,
       contrasts::AbstractDict{Symbol}=Dict{Symbol,Any}())
    lm(X::AbstractMatrix, y::AbstractVector;
       wts::AbstractVector=similar(y, 0), dropcollinear::Bool=true, method::Symbol=:cholesky)

Fit a linear model to data.
An alias for `fit(LinearModel, X, y; wts=wts, dropcollinear=dropcollinear, method=method)`

$FIT_LM_DOC
"""
lm(X, y, allowrankdeficient_dep::Union{Bool,Nothing}=nothing; kwargs...) =
    fit(LinearModel, X, y, allowrankdeficient_dep; kwargs...)

dof(x::LinearModel) = linpred_rank(x.pp) + 1

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
    wts = obj.rr.wts

    if hasintercept(obj)
        if isempty(wts)
            m = mean(y)
        else
            m = mean(y, weights(wts))
        end
    else
        @warn("Starting from GLM.jl 1.8, null model is defined as having no predictor at all " *
              "when a model without an intercept is passed.")
        m = zero(eltype(y))
    end

    v = zero(eltype(y))*zero(eltype(wts))
    if isempty(wts)
        @inbounds @simd for yi in y
            v += abs2(yi - m)
        end
    else
        @inbounds @simd for i = eachindex(y,wts)
            v += abs2(y[i] - m)*wts[i]
        end
    end
    v
end

loglikelihood(obj::LinearModel) = loglikelihood(obj.rr)

function nullloglikelihood(obj::LinearModel)
    r = obj.rr
    n = isempty(r.wts) ? length(r.y) : sum(r.wts)
    -n/2 * (log(2π * nulldeviance(obj)/n) + 1)
end

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
    cn = coefnames(mm)
    CoefTable(hcat(cc,se,tt,p,cc+ci,cc-ci),
              ["Coef.","Std. Error","t","Pr(>|t|)","Lower $levstr%","Upper $levstr%"],
              cn, 4, 3)
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
                 interval::Union{Symbol,Nothing}=nothing, level::Real=0.95)
    retmean = similar(view(newx, :, 1))
    if interval === nothing
        res = retmean
        predict!(res, mm, newx)
    else
        res = (prediction=retmean, lower=similar(retmean), upper=similar(retmean))
        predict!(res, mm, newx, interval=interval, level=level)
    end
    return res
end

function StatsModels.predict!(res::Union{AbstractVector,
                                         NamedTuple{(:prediction, :lower, :upper),
                                                    <:NTuple{3, AbstractVector}}},
                              mm::LinearModel, newx::AbstractMatrix;
                              interval::Union{Symbol, Nothing}=nothing,
                              level::Real=0.95)
    if interval === :confint
        Base.depwarn("interval=:confint is deprecated in favor of interval=:confidence", :predict)
        interval = :confidence
    end
    if interval === nothing
        res isa AbstractVector ||
            throw(ArgumentError("`res` must be a vector when `interval == nothing` or is omitted"))
        length(res) == size(newx, 1) ||
            throw(DimensionMismatch("length of `res` must equal the number of rows in `newx`"))
        res .= newx * coef(mm)
    elseif mm.pp isa DensePredChol && 
        mm.pp.chol isa CholeskyPivoted &&
        mm.pp.chol.rank < size(mm.pp.chol, 2)
            throw(ArgumentError("prediction intervals are currently not implemented " *
                                "when some independent variables have been dropped " *
                                "from the model due to collinearity"))
    elseif mm.pp isa DensePredQR && rank(mm.pp.qr.R) < size(mm.pp.qr.R, 2)
        throw(ArgumentError("prediction intervals are currently not implemented " *
                            "when some independent variables have been dropped " *
                            "from the model due to collinearity"))
    else
        res isa NamedTuple ||
            throw(ArgumentError("`res` must be a `NamedTuple` when `interval` is " *
                                "`:confidence` or `:prediction`"))
        prediction, lower, upper = res
        length(prediction) == length(lower) == length(upper) == size(newx, 1) ||
            throw(DimensionMismatch("length of vectors in `res` must equal the number of rows in `newx`"))
        length(mm.rr.wts) == 0 || error("prediction with confidence intervals not yet implemented for weighted regression")

        dev = deviance(mm)
        dofr = dof_residual(mm)
        ret = diag(newx*vcov(mm)*newx')
        if interval == :prediction
            ret .+= dev/dofr
        elseif interval != :confidence
            error("only :confidence and :prediction intervals are defined")
        end
        ret .= quantile(TDist(dofr), (1 - level)/2) .* sqrt.(ret)
        prediction .= newx * coef(mm)
        lower .= prediction .+ ret
        upper .= prediction -+ ret
    end
    return res
end

function confint(obj::LinearModel; level::Real=0.95)
    hcat(coef(obj),coef(obj)) + stderror(obj) *
    quantile(TDist(dof_residual(obj)), (1. - level)/2.) * [1. -1.]
end

"""
    cooksdistance(obj::LinearModel)

Compute [Cook's distance](https://en.wikipedia.org/wiki/Cook%27s_distance)
for each observation in linear model `obj`, giving an estimate of the influence
of each data point.
Currently only implemented for linear models without weights.
"""
function StatsBase.cooksdistance(obj::LinearModel)
    u = residuals(obj)
    mse = dispersion(obj,true)
    k = dof(obj)-1
    d_res = dof_residual(obj)
    X = modelmatrix(obj)
    XtX = crossmodelmatrix(obj)
    k == size(X,2) || throw(ArgumentError("Models with collinear terms are not currently supported."))
    wts = obj.rr.wts
    if isempty(wts)
        hii = diag(X * inv(XtX) * X')
    else
        throw(ArgumentError("Weighted models are not currently supported."))
    end
    D = @. u^2 * (hii / (1 - hii)^2) / (k*mse)
    return D
end
