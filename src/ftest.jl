type FTestResult
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

function issubmodel(mod1::LinPredModel, mod2::LinPredModel)
    mod1.rr.y != mod2.rr.y && return false # Response variables must be equal

    # Now, test that all predictor variables are equal
    mod1_pred = mod1.pp.X
    mod1_npreds = size(mod1_pred, 2)
    mod2_pred = mod2.pp.X
    mod2_npreds = size(mod1_pred, 2)
    # If model 1 has more predictors, it can't possibly be a submodel
    mod1_npreds > mod2_npreds && return false 
    
    for i in 1:mod1_npreds
        var_in_mod2 = false
        for j in 1:mod2_npreds
            if mod1_pred[:, i] == mod2_pred[:, j]
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
    df1 = dof_residual(mod1)

    MSR1 = (SSR2-SSR1)/df2
    MSR2 = SSR1/df1

    fstat = MSR1/MSR2
    pval = ccdf(FDist(df2, df1), fstat)

    return FTestResult(SSR1, SSR2, df1, df2, MSR1, MSR2, fstat, pval)
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

    return CoefTable([SSR1s, SSR2s, df1s, df2s, MSR1s, MSR2s, fstats, pvals],
    ["Model 1 SSR", "Model 2 SSR", "Model 1 df", "Model 2 df", "Model 1 MSR", "Model 2 MSR", "F statistic", "p-value"],
    ["Model $(i-1):$i" for i in 2:nmodels])
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

export show
