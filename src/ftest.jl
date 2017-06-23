type FTestResult{N}
    ssr::NTuple{N, Float64}
    nparams::NTuple{N, Int}
    dof_resid::NTuple{N, Int}
    r2::NTuple{N, Float64}
    fstat::Tuple{Vararg{Float64}}
    pval::Tuple{Vararg{PValue}}
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

_diffn{N, T}(t::NTuple{N, T})::NTuple{N, T} =  ntuple(i->t[i]-t[i+1], N-1)

_diff{N, T}(t::NTuple{N, T})::NTuple{N, T} =  ntuple(i->t[i+1]-t[i], N-1)

import Base: ./
./{N, T1, T2}(t1::NTuple{N, T1}, t2::NTuple{N, T2}) = ntuple(i->t1[i]/t2[i], N)

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

    SSR = deviance.(mods)

    nparams = dof.(mods)

    df2 = _diffn(nparams)
    df1 = Int.(dof_residual.(mods))

    MSR1 = _diff(SSR)./df2
    MSR2 = (SSR./df1)[1:nmodels-1]

    fstat = MSR1./MSR2
    pval = PValue.(ccdf.(FDist.(df2, df1[1:nmodels-1]), fstat))
    return FTestResult(SSR, nparams, df1, r2.(mods), fstat, pval)
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

function show{N}(io::IO, ftr::FTestResult{N})
    Δdof = _diffn(ftr.dof_resid)
    Δssr = _diffn(ftr.ssr)
    ΔR² = _diffn(ftr.r2)

    nc = 10
    nr = N
    outrows = Array{String, 2}(nr+1, nc)
    
    outrows[1, :] = ["", "Res. DOF",  "DOF",  "ΔDOF",  "SSR",
                    "ΔSSR",  "R²",  "ΔR²",  "F*",  "p(>F)"]
    outrows[2, :] = ["Model 1", @sprintf("%.4f", ftr.dof_resid[1]),
                     @sprintf("%.4f", ftr.nparams[1]), " ", @sprintf("%.4f", ftr.ssr[1]),
                     " ", @sprintf("%.4f", ftr.r2[1]), " ", " ", " "]
    
    for i in 2:nr
        outrows[i+1, :] = ["Model $i", @sprintf("%.4f", ftr.dof_resid[i]),
                           @sprintf("%.4f", ftr.nparams[i]), @sprintf("%.4f",
                           Δdof[i-1]), @sprintf("%.4f", ftr.ssr[i]),
                           @sprintf("%.4f", Δssr[i-1]), @sprintf("%.4f", ftr.r2[i]),
                           @sprintf("%.4f", ΔR²[i-1]), @sprintf("%.4f", ftr.fstat[i-1]),
                           string(ftr.pval[i-1])]
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


