type FTestResult
    mod1::RegressionModel
    mod2::RegressionModel
    SSR1::Float64 # sum of squared residuals, model 1
    SSR2::Float64 # sum of squared residuals, model 2
    df1::Int # degrees of freedom, model 1 residuals
    df2::Int # degrees of freedom, model 2 residuals
    MSR1::Float64 # mean of squared residuals, model 1
    MSR2::Float64 # mean of squared residuals, model 2
    fstat::Float64 # f statistic
    pval::Float64 # p value
end

type MultiFTestResult
    results::Array{FTestResult, 1}
end

function ftest(mod1::RegressionModel, mod2::RegressionModel)
    SSR1 = deviance(mod1.model.rr)
    SSR2 = deviance(mod2.model.rr)

    nparams1 = length(mod1.model.pp.beta0)
    nparams2 = length(mod2.model.pp.beta0)

    df2 = nparams1-nparams2
    df1 = dof_residual(mod1)

    MSR1 = (SSR2-SSR1)/df2
    MSR2 = SSR1/df1

    fstat = MSR1/MSR2
    pval = ccdf(FDist(df2, df1), fstat)

    return FTestResult(mod1, mod2, SSR1, SSR2, df1, df2, MSR1, MSR2, fstat, pval)
end

function ftest(mods::RegressionModel...)
    nmodels = length(mods)
    results = Array{FTestResult, 1}((nmodels^2)-nmodels)
    
    idx = 1
    for (i, mod1) in enumerate(mods)
	for (j, mod2) in enumerate(mods)
	    i == j && continue
	    results[idx] = ftest(mod1, mod2)
	    idx += 1
	end
    end

    return MultiFTestResult(results)
end

### Utility functions to show FTestResult and MultiFTestResult
function show(io::IO, x::FTestResult)
    if get(io, :compact, true)
        lines = [string("Fisher 2-model F test with ", x.df1, " and ", x.df2, " degrees of freedom"), #=string("Model 1: ", formula(x.mod1), " Model 2: ", formula(x.mod2)),=# string("Sums of squared residuals: ", x.SSR1, " and ", x.SSR2), string("F* = ", x.fstat, " p = ", x.pval)]
	outlen = maximum(length, lines)
    
	println(io, RepString("=", outlen))
        for line in lines
	   println(io, line)
	end
    else
	print(io, "p = ", x.pval)
    end
end

function show(io::IO, x::MultiFTestResult)
    for res in x.results
	show(io, res)
    end
end

export ftest
export show
