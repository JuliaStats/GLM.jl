module PlotsRecipes

 # Plot functions
 export cooksleverageplot, cooksleverageplot!
 export scalelocationplot, scalelocationplot!
 export residualplot, residualplot!
 export residualsleverageplot, residualsleverageplot!
 export lmplot


"""
    lmplot(obj::LinearModel; kw...)

Display several summary plots of a linear model.

Keyword arguments for the plotting backend such as `size` are supported. If using Makie, only keyword arguments to `Figure` are supported.

## Examples
```julia-repl
julia> using GLM, StatsPlots, GLM.PlotsRecipes

julia> X = randn(30, 5); y = X * randn(5) + 0.3*randn(30)

julia> l = lm(X,y)

julia> lmplot(l)
```
"""
function lmplot end

"""
    cooksleverageplot(obj::LinearModel; kw...)

Plot the Cook's distances of a linear model against its leverages.

Keyword arguments are passed to the plotting backend.
"""
function cooksleverageplot end
function cooksleverageplot! end

"""
    scalelocationplot(obj::LinearModel, kw...)

Plot the root standardized residuals of a linear model against its fitted values.

Keyword arguments are passed to the plotting backend.
"""
function scalelocationplot end
function scalelocationplot! end

"""
    residualplot(obj::LinearModel, kw...)

Plot the residuals of a linear model against its fitted values.

## keyword arguments

* `axislines = true` whether to display a line on the x axis.
* `axislinecolor`
* `axislinestyle`
* `axislinewidth`

Other keyword arguments are passed to the plotting backend.
"""
function residualplot end
function residualplot! end

"""
    residualsleverageplot(obj::LinearModel, kw...)

Plot the residuals of a linear model against its leverages.

## keyword arguments

* `axislines = true`
* `axislinecolor`
* `axislinestyle`
* `axislinewidth`
* `cookslevels = [0.5,2.0]` Levels curves of Cook's distance to display.
* `cookslinecolor`
* `cookslinestyle`
* `cookslinewidth`

Other keyword arguments are passed to the plotting backend.
"""
function residualsleverageplot end
function residualsleverageplot! end

end
