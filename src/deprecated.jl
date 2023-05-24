@deprecate predict(mm::LinearModel, newx::AbstractMatrix, interval::Symbol, level::Real = 0.95) predict(mm, newx; interval=interval, level=level)

@deprecate confint(obj::LinearModel, level::Real) confint(obj, level=level)
@deprecate confint(obj::AbstractGLM, level::Real) confint(obj, level=level)

@deprecate installbeta!(p) (p.beta0 .= p.delbeta) false
@deprecate installbeta!(p, f) (p.beta0 .+= p.delbeta .* f) false
