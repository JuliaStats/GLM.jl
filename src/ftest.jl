type FTestResult
    ntests::Int
    SSR::Array{Float64, 1}
    nparams::Array{Int, 1}
    dof_resid::Array{Int, 1}
    R2::Array{Float64, 1}
    fstat::Array{Float64, 1}
    pval::Array{PValue, 1}
end

#function FTestResult{N}(SSR1::Array{Float64, 1}, fstati
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
    ftest(mod::LinearModel...)

For each sequential pair of linear predictors in `mod`, perform an F-test to determine if 
the first one fits significantly better than the second.

A table is returned containing residual degrees of freedom (DOF), degrees of freedom,
difference in DOF from the preceding model, sum of squared residuals (SSR), difference in
SSR from the preceding model, R², difference in R² from the preceding model, and F 
statistic and p-value for the comparison between the two models.

Note: This function can be used to do an ANOVA, by testing the relative fit of two models
to the data

# Examples:
Suppose we want to compare the effects of two or more treatments on some result. Because
this is an ANOVA, our null hypothesis is that Result~1 fits the data as well as
Result~Treatment. 
```jldoctest
julia> dat = DataFrame(Treatment=[1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2.],
                     Result=[1.1, 1.2, 1, 2.2, 1.9, 2, .9, 1, 1, 2.2, 2, 2]);
julia> mod = lm(@formula(Result~Treatment), dat        );
julia> nullmod = lm(@formula(Result~1), d);
julia> ft = ftest(mod.model, nullmod.model)
         Residual DOF DOF ΔDOF      SSR    ΔSSR           R²       ΔR²      F*      p(>F)
Model 1          10.0 3.0  NaN 0.128333     NaN     0.960258       NaN     NaN        NaN
Model 2          11.0 2.0 -1.0  3.22917 3.10083 -2.22045e-16 -0.960258 241.623 2.48122e-8


```
"""
function ftest(mods::LinearModel...)
    nmodels = length(mods)
    for i in 2:nmodels
        issubmodel(mods[i], mods[i-1]) || 
        throw(ArgumentError("F test $i is only valid if model $i is nested in model $i-1"))
    end

    SSR = collect(deviance.(mods))

    nparams = collect(dof.(mods))

    df2 = -diff(nparams)
    df1 = collect(dof_residual.(mods))

    MSR1 = diff(SSR)./df2
    MSR2 = view(SSR, 1:nmodels-1)./view(df1, 1:nmodels-1)

    fstat = MSR1./MSR2
    pval = PValue.(ccdf.(FDist.(df2, view(df1, 1:nmodels-1)), fstat))
    return FTestResult(nmodels, SSR, nparams, df1, collect(r2.(mods)), fstat, pval)
#=
    prepend!(pval, [NaN])
    prepend!(fstat, [NaN])

    df2 = Float64.(df2)
    prepend!(df2, [NaN])

    R2 = collect(r2.(mods))
    ΔR2 = -diff(R2)
    prepend!(ΔR2, [NaN])

    ΔSSR = -diff(SSR)
    prepend!(ΔSSR, [NaN])
    return CoefTable([df1, nparams, df2, SSR, ΔSSR, R2, ΔR2, fstat, pval],
                     ["Res. DOF", "DOF", "ΔDOF", "SSR", "ΔSSR", "R²", "ΔR²", "F*",
                     "p(>F)"],
                     ["Model $i" for i in 1:nmodels])=#
end

function show(io::IO, ftr::FTestResult)
    ssr = ftr.SSR; nparams = ftr.nparams; dof_resid = ftr.dof_resid
    R² = ftr.R2; fstat = ftr.fstat; pval = ftr.pval

    Δdof = -diff(dof_resid)
    Δssr = -diff(ssr)
    ΔR² = -diff(R²)

    nc = 10
    nr = ftr.ntests
    outrows = Array{String, 2}(nr+1, nc)
    
    outrows[1, :] = ["", "Res. DOF",  "DOF",  "ΔDOF",  "SSR",  "ΔSSR",  "R²",  "ΔR²",  "F*",  "p(>F)"]
    outrows[2, :] = ["Model 1", @sprintf("%.4f", dof_resid[1]), @sprintf("%.4f", nparams[1]), " ", @sprintf("%.4f", ssr[1]), " ", @sprintf("%.4f", R²[1]), " ", " ", " "]
    
    for i in 2:nr
        outrows[i+1, :] = ["Model $i", @sprintf("%.4f", dof_resid[i]), @sprintf("%.4f", nparams[i]), @sprintf("%.4f", Δdof[i-1]), @sprintf("%.4f", ssr[i]), @sprintf("%.4f", Δssr[i-1]), @sprintf("%.4f", R²[i]), @sprintf("%.4f", ΔR²[i-1]), @sprintf("%.4f", fstat[i-1]), string(pval[i-1])]
    end
    colwidths = length.(outrows)
    max_colwidths = [maximum(view(colwidths, :, i)) for i in 1:nc]

    for r in 1:nr+1
        for c in 1:nc
            cur_cell = outrows[r, c]
            cur_cell_len = length(cur_cell)
            
            print(io, cur_cell)
            print(io, RepString(" ", max_colwidths[c]-cur_cell_len+1))
        end
        print(io, "\n")
    end
end


