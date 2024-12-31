
"""
    standardized_residuals(obj::LinearModel)

Compute the standardized residuals of a linear model, defined for the `i`-th observation as
```
r[i] / (std(r) * sqrt(1 - h[i]),
```
where `r` are the residuals of the model, ``s`` is the empirical standard deviation of the residuals and ``h[i]`` is the leverage of observation `i`.
"""
function standardized_residuals(obj::LinearModel)
    r = residuals(obj)
    h = leverage(obj)
    return r ./(std(r) .* sqrt.(1 .- h))
end

function lmplot end

function cooksleverageplot end
function cooksleverageplot! end
function scalelocationplot end
function scalelocationplot! end
function residualplot end
function residualplot! end
function residualsleverageplot end
function residualsleverageplot! end
function quantilequantileplot end
function quantilequantileplot! end
