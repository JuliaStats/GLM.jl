@deprecate predict(mm::LinearModel, newx::AbstractMatrix, interval::Symbol, level::Real = 0.95) predict(mm, newx; interval=interval, level=level)

@deprecate confint(obj::LinearModel, level::Real) confint(obj, level=level)
@deprecate confint(obj::AbstractGLM, level::Real) confint(obj, level=level)

function getproperty(m::AbstractGLM, field)
    if field === :model
        Base.depwarn("accessing the `model` field of `AbstractGLM` objects is deprecated, " *
                     "as it now returns its parent object", :getproperty)
        return m
    else
        return getfield(m, field)
    end
end