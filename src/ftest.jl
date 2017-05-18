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


function ftest(mod1::LinPredModel, mod2::LinPredModel)
    @argcheck issubmodel(mod2, mod1) "F test is only valid if model 2 is nested in model 1"
    SSR1 = deviance(mod1)
    SSR2 = deviance(mod2)

    nparams1 = dof(mod1)
    nparams2 = dof(mod2)

    df2 = nparams1-nparams2
    df1 = Int(dof_residual(mod1))

    MSR1 = (SSR2-SSR1)/df2
    MSR2 = SSR1/df1

    fstat = MSR1/MSR2
    pval = ccdf(FDist(df2, df1), fstat)

    return FTestTable(SSR1, SSR2, df1, df2, MSR1, MSR2, fstat, pval)
end

function ftest(mods::LinPredModel...)
    nmodels = length(mods)
    SSR1s = Array{FTestResult, 1}(nmodels-1)
    SSR2s = similar(SSR1s)
    df1s = similar(SSR1s)
    df2s = similar(SSR1s)
    MSR1s = similar(SSR1s)
    MSR2s = similar(SSR1s)
    fstats = similar(SSR1s)
    pvals = similar(SSR1s)
    
    for (i, mod1) in enumerate(mods[2:end])
        result = ftest(mods[i], mod1)
        SSR1s[i] = result.SSR1
        SSR2s[i] = result.SSR2
        df1s[i] = result.df1
        df2s[i] = result.df2
        MSR1s[i] = result.MSR1
        MSR2s[i] = result.MSR2
        fstats[i] = result.fstat
        pvals[i] = result.pval
    end

    return FTestTable(SSR1s, SSR2s, df1s, df2s, MSR1s, MSR2s, fstats, pvals)
end

### Utility functions to show FTestResult and MultiFTestResult
function show(io::IO, x::FTestTable)
    if get(io, :compact, true)
        for i in 1:x.ntests
	    lines = [string("Fisher 2-model F test with ", x.df1[i], " and ", x.df2[i], " degrees of freedom"), string("Sums of squared residuals: ", x.SSR1[i], " and ", x.SSR2[i]), string("F* = ", x.fstat[i], " p = ", x.pval[i])]
	    outlen = maximum(length, lines)

	    println(io, RepString("=", outlen))
	    for line in lines
                println(io, line)
            end
	end
    else
	for i in 1:x.ntests
            print(io, "p = ", x.pval)
	end
    end
end

export show
