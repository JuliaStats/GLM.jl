@deprecate predict(mm::LinearModel, newx::AbstractMatrix, interval::Symbol, level::Real = 0.95) predict(mm, newx; interval=interval, level=level)

@deprecate confint(obj::LinearModel, level::Real) confint(obj, level=level)
@deprecate confint(obj::AbstractGLM, level::Real) confint(obj, level=level)

function Base.getproperty(mm::LinPredModel, f::Symbol)
    if f === :model
        Base.depwarn("accessing the `model` field of GLM.jl models is deprecated, " *
                     "as they are no longer wrapped in a `TableRegressionModel` " *
                     "and can be used directly now", :getproperty)
        return mm
    else
        return getfield(mm, f)
    end
end