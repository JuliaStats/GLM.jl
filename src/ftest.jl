mutable struct FTestResult{N}
    ssr::NTuple{N, Float64}
    dof::NTuple{N, Int}
    dof_resid::NTuple{N, Int}
    r2::NTuple{N, Float64}
    fstat::Tuple{Vararg{Float64}}
    pval::Tuple{Vararg{PValue}}
end

"""A helper function to determine if mod1 is nested in mod2"""
function issubmodel(mod1::LinPredModel, mod2::LinPredModel; atol=0::Real)
    mod1.rr.y != mod2.rr.y && return false # Response variables must be equal

    # Test that models are nested
    pred1 = mod1.pp.X
    npreds1 = size(pred1, 2)
    pred2 = mod2.pp.X
    npreds2 = size(pred2, 2)
    # If model 1 has more predictors, it can't possibly be a submodel
    npreds1 > npreds2 && return false
    # Test min norm pred2*B - pred1 ≈ 0
    rtol = Base.rtoldefault(typeof(pred1[1,1]))
    nresp = size(pred2, 1)
    return norm(view(qr(pred2).Q'pred1, npreds2 + 1:nresp, :)) <= max(atol, rtol*norm(pred1))
end

_diffn(t::NTuple{N, T}) where {N, T} = ntuple(i->t[i]-t[i+1], N-1)

_diff(t::NTuple{N, T}) where {N, T} = ntuple(i->t[i+1]-t[i], N-1)

dividetuples(t1::NTuple{N}, t2::NTuple{N}) where {N} = ntuple(i->t1[i]/t2[i], N)

"""
    ftest(mod::LinearModel...; atol=0::Real)

For each sequential pair of linear predictors in `mod`, perform an F-test to determine if
the first one fits significantly better than the next.

A table is returned containing residual degrees of freedom (DOF), degrees of freedom,
difference in DOF from the preceding model, sum of squared residuals (SSR), difference in
SSR from the preceding model, R², difference in R² from the preceding model, and F-statistic
and p-value for the comparison between the two models.

!!! note
    This function can be used to perform an ANOVA by testing the relative fit of two models
    to the data

Optional keyword argument `atol` controls the numerical tolerance when testing whether
the models are nested.

# Examples

Suppose we want to compare the effects of two or more treatments on some result. Because
this is an ANOVA, our null hypothesis is that `Result ~ 1` fits the data as well as
`Result ~ 1 + Treatment`.

```jldoctest
julia> dat = DataFrame(Treatment=[1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2.],
                       Result=[1.1, 1.2, 1, 2.2, 1.9, 2, .9, 1, 1, 2.2, 2, 2],
                       Other=[1, 1, 2, 1, 2, 1, 3, 1, 1, 2, 2, 1]);

julia> model = lm(@formula(Result ~ 1 + Treatment), dat);

julia> nullmodel = lm(@formula(Result ~ 1), dat);

julia> bigmodel = lm(@formula(Result ~ 1 + Treatment + Other), dat);

julia> ft = ftest(model.model, nullmodel.model)
        Res. DOF DOF ΔDOF    SSR    ΔSSR     R²    ΔR²       F* p(>F)
Model 1       10   3      0.1283         0.9603
Model 2       11   2   -1 3.2292 -3.1008 0.0000 0.9603 241.6234 <1e-7


julia> ftest(bigmodel.model, model.model, nullmodel.model)
        Res. DOF DOF ΔDOF    SSR    ΔSSR     R²    ΔR²       F*  p(>F)
Model 1        9   4      0.1038         0.9678
Model 2       10   3   -1 0.1283 -0.0245 0.9603 0.0076   2.1236 0.1790
Model 3       11   2   -1 3.2292 -3.1008 0.0000 0.9603 241.6234  <1e-7
```
"""
function ftest(mods::LinearModel...; atol=0::Real)
    nmodels = length(mods)
    for i in 2:nmodels
        issubmodel(mods[i], mods[i-1], atol=atol) ||
        throw(ArgumentError("F test $i is only valid if model $i is nested in model $(i-1)"))
    end

    SSR = deviance.(mods)

    nparams = dof.(mods)

    df2 = _diffn(nparams)
    df1 = Int.(dof_residual.(mods))

    MSR1 = dividetuples(_diff(SSR), df2)
    MSR2 = dividetuples(SSR, df1)[1:nmodels-1]

    fstat = dividetuples(MSR1, MSR2)
    pval = PValue.(ccdf.(FDist.(df2, df1[1:nmodels-1]), fstat))
    return FTestResult(SSR, nparams, df1, r2.(mods), fstat, pval)
end

function show(io::IO, ftr::FTestResult{N}) where N
    Δdof = _diffn(ftr.dof_resid)
    Δssr = _diffn(ftr.ssr)
    ΔR² = _diffn(ftr.r2)

    nc = 10
    nr = N
    outrows = Matrix{String}(undef, nr+1, nc)

    outrows[1, :] = ["", "Res. DOF", "DOF", "ΔDOF", "SSR", "ΔSSR",
                     "R²", "ΔR²", "F*", "p(>F)"]

    outrows[2, :] = ["Model 1", @sprintf("%.0d", ftr.dof_resid[1]),
                     @sprintf("%.0d", ftr.dof[1]), " ",
                     @sprintf("%.4f", ftr.ssr[1]), " ",
                     @sprintf("%.4f", ftr.r2[1]), " ", " ", " "]

    for i in 2:nr
        outrows[i+1, :] = ["Model $i", @sprintf("%.0d", ftr.dof_resid[i]),
                           @sprintf("%.0d", ftr.dof[i]), @sprintf("%.0d", Δdof[i-1]),
                           @sprintf("%.4f", ftr.ssr[i]), @sprintf("%.4f", Δssr[i-1]),
                           @sprintf("%.4f", ftr.r2[i]), @sprintf("%.4f", ΔR²[i-1]),
                           @sprintf("%.4f", ftr.fstat[i-1]), string(ftr.pval[i-1]) ]
    end
    colwidths = length.(outrows)
    max_colwidths = [maximum(view(colwidths, :, i)) for i in 1:nc]

    for r in 1:nr+1
        for c in 1:nc
            cur_cell = outrows[r, c]
            cur_cell_len = length(cur_cell)

            padding = " "^(max_colwidths[c]-cur_cell_len)
            if c > 1
                padding = " "*padding
            end

            print(io, padding)
            print(io, cur_cell)
        end
        print(io, "\n")
    end
end
