type FTestTable
    ntests::Int
    SSR1::Array{Float64, 1} # sum of squared residuals, model 1
    SSR2::Array{Float64, 1} # sum of squared residuals, model 2
    df1::Array{Int, 1} # degrees of freedom, model 1 residuals
    df2::Array{Int, 1} # degrees of freedom, model 2 residuals
    MSR1::Array{Float64, 1} # mean of squared residuals, model 1
    MSR2::Array{Float64, 1} # mean of squared residuals, model 2
    fstat::Array{Float64, 1} # f statistic
    pval::Array{Float64, 1} # p value
end

function FTestTable(SSR1::Array{Float64, 1}, SSR2::Array{Float64, 1}, df1::Array{Int, 1}, df2::Array{Int, 1}, MSR1::Array{Float64, 1}, MSR2::Array{Float64, 1}, fstat::Array{Float64, 1}, pval::Array{Float64, 1})
#Note: Is checking args in internal helper functions the desired behavior?
@argcheck length(SSR1) == length(SSR2) == length(df1) == length(df2) == length(MSR1) == length(MSR2) == length(fstat) == length(pval) "F test values  must all be the same length"

    return FTestTable(length(SSR1), SSR1, SSR2, df1, df2, MSR1, MSR2, fstat, pval)
end

function FTestTable(SSR1::Float64, SSR2::Float64, df1::Int, df2::Int, MSR1::Float64, MSR2::Float64, fstat::Float64, pval::Float64)
    return  FTestTable([SSR1], [SSR2], [df1], [df2], [MSR1], [MSR2], [fstat], [pval])
end

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
    `function ftest(mod::LinPredModel...)`
Test if mod[i] fits significantly better than mod[i+1]

Note: This function can easily be used to do an ANOVA. ANOVA is nothing more than a test
to see if one model fits the data better than another.

# Examples:
As usual, we want to compare a result across two or more treatments. In the classic ANOVA
framework, our null hypothesis is that Result~1 is a perfectly good fit to the data. 
The alternative for ANOVA is that Result~Treatment is a better fit to the data than Result~1
```jldoctest
julia> d = DataFrame(Treatment=[1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2.], Result=[1.1, 1.2, 1, 2.2, 1.9, 2, .9, 1, 1, 2.2, 2, 2])
julia> mod = lm(Result~Treatment, d)
julia> nullmod = lm(Result~1, d)
julia> ft = ftest(mod.model, nullmod.model)
====================================================================
Fisher 2-model F test with 10 and 1 degrees of freedom
Sums of squared residuals: 0.12833333333333344 and 3.229166666666667
F* = 241.62337662337643 p = 2.481215056713184e-8
```
"""
function ftest(mods::LinPredModel...)
    nmodels = length(mods)
    for i in 2:nmodels
        @argcheck issubmodel(mods[i], mods[i-1]) "F test is only valid if model 2 is nested in model 1"
    end

    SSR1 = [deviance(mods[i]) for i in 1:nmodels-1]
    SSR2 = [deviance(mods[i]) for i in 2:nmodels]

    nparams1 = [dof(mods[i]) for i in 1:nmodels-1]
    nparams2 = [dof(mods[i]) for i in 2:nmodels]

    df2 = nparams1-nparams2
    df1 = [Int(dof_residual(mods[i])) for i in 1:nmodels-1]

    MSR1 = (SSR2-SSR1)./df2
    MSR2 = SSR1./df1

    fstat = MSR1./MSR2
    pval = ccdf.(FDist.(df2, df1), fstat)

    return FTestTable(SSR1, SSR2, df1, df2, MSR1, MSR2, fstat, pval)
end

### Utility functions to show FTestResult and MultiFTestResult
function show(io::IO, x::FTestTable)
    if get(io, :compact, true)
#=        if x.ntests == 1
            lines = [string("Fisher 2-model F test with ", x.df1[1], " and ", x.df2[1], " degrees of freedom"), string("Sums of squared residuals: ", round(x.SSR1[1], 6), " and ", round(x.SSR2[1], 6)), string("F* = ", round(x.fstat[1], 6), " p = ", PValue(x.pval[1]))]
            outlen = maximum(length, lines)

            println(io, RepString("=", outlen))
            for line in lines
                println(io, line)
            end
        else=#
	    nequals = 86
            println(io, "Fisher 2-model F tests (between ajdacent pairs of models)")
            print(io, "\n")
            println(io, RepString("=", nequals))
            println(io, "Comparison\tDF1\tDF2\tSSR1\t\tSSR2\t\tF*\t\tp(>F)")
            println(io, RepString("-", nequals))
            for i in 1:x.ntests
                println(io, i, ":", i+1, "\t\t", x.df1[i], "\t", x.df2[i], "\t", round(x.SSR1[i], 6), "\t", round(x.SSR2[i], 6), "\t", round(x.fstat[i], 6), "\t", PValue(x.pval[i]))
            end
            println(io, RepString("=", nequals))
#        end
        
    else
        for i in 1:x.ntests
            print(io, "p = ", x.pval)
        end
    end
end

export show
