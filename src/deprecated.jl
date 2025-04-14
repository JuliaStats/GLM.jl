function Base.getproperty(mm::LinPredModel, f::Symbol)
    if f === :model
        Base.depwarn("accessing the `model` field of GLM.jl models is deprecated, " *
                     "as they are no longer wrapped in a `TableRegressionModel` " *
                     "and can be used directly now", :getproperty)
        return mm
    elseif f === :mf
        Base.depwarn("accessing the `mf` field of GLM.jl models is deprecated, " *
                     "as they are no longer wrapped in a `TableRegressionModel`." *
                     "Use `formula(m)` to access the model formula.", :getproperty)
        form = formula(mm)
        return ModelFrame{Nothing,typeof(mm)}(form, nothing, nothing, typeof(mm))
    else
        return getfield(mm, f)
    end
end
