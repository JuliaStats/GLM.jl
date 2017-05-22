"""A helper function to determine if mod1 is nested in mod2"""
function issubmodel(mod1::LinPredModel, mod2::LinPredModel)
    mod1.rr.y != mod2.rr.y && return false # Response variables must be equal

    # Now, test that all predictor variables are equal
    pred1 = mod1.pp.X
    npreds1 = size(pred1, 2)
    pred2 = mod2.pp.X
    npreds2 = size(pred1, 2)
    # If model 1 has more predictors, it can't possibly be a submodel
    npreds1 > npreds2 && return false 
    
    @inbounds for i in 1:npreds1
        var_in_mod2 = false
        for j in 1:npreds2
            if view(pred1, :, i) == view(pred2, :, j)
                var_in_mod2 = true
                break
            end
        end
        if !var_in_mod2
            # We have found a predictor variable in model 1 that is not in model 2
            return false 
        end
    end

    return true
end

"""
    ftest(mod::LinPredModel...)

For each sequential pair of linear predictors in `mod`, perform an F-test to determine if the first one fits significantly better than the second

Note: This function can easily be used to do an ANOVA. ANOVA is nothing more than a test
to see if one model fits the data better than another.

# Examples:
As usual, we want to compare a result across two or more treatments. In the classic ANOVA
framework, our null hypothesis is that Result~1 is a perfectly good fit to the data. 
The alternative for ANOVA is that Result~Treatment is a better fit to the data than Result~1
```jldoctest
julia> d = DataFrame(Treatment=[1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2.], Result=[1.1, 1.2, 1, 2.2, 1.9, 2, .9, 1, 1, 2.2, 2, 2]);
julia> mod = lm(@formula(Result~Treatment), d);
julia> nullmod = lm(@formula(Result~1), d);
julia> ft = ftest(mod.model, nullmod.model)
         Residual DOF ΔResidual DOF DOF Δdof      SSR    ΔSSR           R²       ΔR²      F*      p(>F)
Model 1          10.0           NaN 3.0  NaN 0.128333     NaN     0.960258       NaN     NaN        NaN
Model 2          11.0           1.0 2.0 -1.0  3.22917 3.10083 -2.22045e-16 -0.960258 241.623 2.48122e-8


```
"""
function ftest(mods::LinPredModel...)
    nmodels = length(mods)
    for i in 2:nmodels
        issubmodel(mods[i], mods[i-1]) || 
        throw(ArgumentError("F test is only valid if model 2 is nested in model 1"))
    end

    SSR = collect(deviance.(mods))

    nparams = collect(dof.(mods))

    df2 = -diff(nparams)
    df1 = collect(dof_residual.(mods))

    MSR1 = diff(SSR)./df2
    MSR2 = view(SSR, 1:nmodels-1)./view(df1, 1:nmodels-1)

    fstat = MSR1./MSR2
    pval = ccdf.(FDist.(df2, view(df1, 1:nmodels-1)), fstat)

    prepend!(pval, [NaN])
    prepend!(fstat, [NaN])

    df2 = Float64.(df2)
    prepend!(df2, [NaN])

    Δdf1 = diff(df1)
    prepend!(Δdf1, [NaN])

    R2 = collect(r2.(mods))
    ΔR2 = diff(R2)
    prepend!(ΔR2, [NaN])

    ΔSSR = diff(SSR)
    prepend!(ΔSSR, [NaN])
    return CoefTable([df1, Δdf1, nparams, -df2, SSR, ΔSSR, R2, ΔR2, fstat, pval],
                     ["Residual DOF", "ΔResidual DOF", "DOF", "Δdof", "SSR", "ΔSSR", "R²", "ΔR²", "F*", "p(>F)"],
                     ["Model $i" for i in 1:nmodels])
end
