struct SingleFTestResult
    nobs::Int
    dof::Int
    fstat::Float64
    pval::Float64
end

struct FTestResult{N}
    nobs::Int
    ssr::NTuple{N,Float64}
    dof::NTuple{N,Float64}
    fstat::NTuple{N,Float64}
    pval::NTuple{N,Float64}
end

function StatsModels.isnested(mod1::LinPredModel, mod2::LinPredModel; atol::Real=0.0)
    mod1.rr.y != mod2.rr.y && return false # Response variables must be equal

    # Test that models are nested
    pred1 = mod1.pp.X
    npreds1 = size(pred1, 2)
    pred2 = mod2.pp.X
    npreds2 = size(pred2, 2)
    # If model 1 has more predictors, it can't possibly be a submodel
    npreds1 > npreds2 && return false
    # Test min norm pred2*B - pred1 ≈ 0
    rtol = Base.rtoldefault(typeof(pred1[1, 1]))
    nresp = size(pred2, 1)
    return norm(view(qr(pred2).Q'pred1, (npreds2 + 1):nresp, :)) <=
           max(atol, rtol * norm(pred1))
end

_diffn(t::NTuple{N,T}) where {N,T} = ntuple(i -> t[i] - t[i + 1], N - 1)

_diff(t::NTuple{N,T}) where {N,T} = ntuple(i -> t[i + 1] - t[i], N - 1)

"""
    ftest(mod::LinearModel)

Perform an F-test to determine whether model `mod` fits significantly better
than the null model (i.e. which includes only the intercept).

```jldoctest; setup = :(using DataFrames, GLM)
julia> dat = DataFrame(Result=[1.1, 1.2, 1, 2.2, 1.9, 2, 0.9, 1, 1, 2.2, 2, 2],
                       Treatment=[1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2]);


julia> model = lm(@formula(Result ~ 1 + Treatment), dat);


julia> ftest(model)
F-test against the null model:
F-statistic: 241.62 on 12 observations and 1 degrees of freedom, p-value: <1e-07
```
"""
function ftest(mod::LinearModel)
    hasintercept(mod) ||
        throw(ArgumentError("ftest only works for models with an intercept"))
    wts = weights(mod)
    if wts isa ProbabilityWeights
        throw(ArgumentError("`ftest` for probability weighted models is not currently supported."))
    end
    rss = deviance(mod)
    tss = nulldeviance(mod)

    n = Int(nobs(mod))
    p = dof(mod) - 2 # -2 for intercept and dispersion parameter
    fstat = abs(((tss - rss) / rss) * ((n - p - 1) / p))
    fdist = FDist(p, dof_residual(mod))

    return SingleFTestResult(n, p, promote(fstat, ccdf(fdist, fstat))...)
end

"""
    ftest(mod::LinearModel...; atol::Real=0.0)

For each sequential pair of linear models in `mod...`, perform an F-test to determine if
the one model fits significantly better than the other. Models must have been fitted
on the same data, and be nested either in forward or backward direction.

A table is returned containing residual degrees of freedom (dof),
absolute difference in dof from the preceding model, sum of squared residuals (SSR),
absolute difference in SSR from the preceding model, and F-statistic and p-value for the comparison
between the two models.

!!! note
    This function can be used to perform an ANOVA by testing the relative fit of two models
    to the data

Optional keyword argument `atol` controls the numerical tolerance when testing whether
the models are nested.

# Examples

Suppose we want to compare the effects of two or more treatments on some result. Because
this is an ANOVA, our null hypothesis is that `Result ~ 1` fits the data as well as
`Result ~ 1 + Treatment`.

```jldoctest ; setup = :(using CategoricalArrays, DataFrames, GLM)
julia> dat = DataFrame(Result=[1.1, 1.2, 1, 2.2, 1.9, 2, 0.9, 1, 1, 2.2, 2, 2],
                       Treatment=[1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
                       Other=categorical([1, 1, 2, 1, 2, 1, 3, 1, 1, 2, 2, 1]));


julia> nullmodel = lm(@formula(Result ~ 1), dat);


julia> model = lm(@formula(Result ~ 1 + Treatment), dat);


julia> bigmodel = lm(@formula(Result ~ 1 + Treatment + Other), dat);

julia> ftest(nullmodel, model)
F-test: 2 models fitted on 12 observations
───────────────────────────────────────────
   SSR  dof    ΔSSR  Δdof        F*   p(>F)
───────────────────────────────────────────
3.2292   11
0.1283   10  3.1008     1  241.6234  <1e-07
───────────────────────────────────────────

julia> ftest(nullmodel, model, bigmodel)
F-test: 3 models fitted on 12 observations
───────────────────────────────────────────
   SSR  dof    ΔSSR  Δdof        F*   p(>F)
───────────────────────────────────────────
3.2292   11
0.1283   10  3.1008     1  241.6234  <1e-07
0.1017    8  0.0266     2    1.0456  0.3950
───────────────────────────────────────────
```
"""
function ftest(mods::LinearModel...; atol::Real=0.0)
    if !all(==(nobs(mods[1])), nobs.(mods))
        throw(ArgumentError("F test is only valid for models fitted on the same data, " *
                            "but number of observations differ"))
    end
    forward = length(mods) == 1 || dof(mods[1]) <= dof(mods[2])
    if forward
        for i in 2:length(mods)
            if dof(mods[i - 1]) >= dof(mods[i]) ||
               !StatsModels.isnested(mods[i - 1], mods[i]; atol=atol)
                throw(ArgumentError("F test is only valid for nested models"))
            end
        end
    else
        for i in 2:length(mods)
            if dof(mods[i]) >= dof(mods[i - 1]) ||
               !StatsModels.isnested(mods[i], mods[i - 1]; atol=atol)
                throw(ArgumentError("F test is only valid for nested models"))
            end
        end
    end

    SSR = deviance.(mods)

    df = dof_residual.(mods)
    Δdf = abs.(_diff(df))
    MSR1 = _diffn(SSR) ./ Δdf
    MSR2 = (SSR ./ df)
    if forward
        MSR2 = MSR2[2:end]
        dfr_big = df[2:end]
    else
        MSR2 = MSR2[1:(end - 1)]
        dfr_big = df[1:(end - 1)]
    end

    fstat = abs.((NaN, (MSR1 ./ MSR2)...))
    pval = (NaN, ccdf.(FDist.(Δdf, dfr_big), fstat[2:end])...)

    return FTestResult(Int(nobs(mods[1])), SSR, float.(df), fstat, pval)
end

function show(io::IO, ftr::SingleFTestResult)
    print(io, "F-test against the null model:\nF-statistic: ",
          StatsBase.TestStat(ftr.fstat), " ")
    print(io, "on ", ftr.nobs, " observations and ", ftr.dof, " degrees of freedom, ")
    return print(io, "p-value: ", PValue(ftr.pval))
end

function show(io::IO, ::MIME"text/plain", ftr::FTestResult{N}) where {N}
    Δdof = abs.(_diff(ftr.dof))
    Δssr = abs.(_diff(ftr.ssr))

    nc = 6
    nr = N
    outrows = Matrix{String}(undef, nr + 1, nc)

    outrows[1, :] = ["SSR", "dof", "ΔSSR", "Δdof", "F*", "p(>F)"]

    outrows[2, :] = [@sprintf("%.4f", ftr.ssr[1]),
                     @sprintf("%.0d", ftr.dof[1]),
                     " ",
                     " ",
                     " ",
                     " "]

    for i in 2:nr
        outrows[i + 1, :] = [@sprintf("%.4f", ftr.ssr[i]),
                             @sprintf("%.0d", ftr.dof[i]),
                             @sprintf("%.4f", Δssr[i - 1]),
                             @sprintf("%.0d", Δdof[i - 1]),
                             @sprintf("%.4f", ftr.fstat[i]),
                             string(PValue(ftr.pval[i]))]
    end
    colwidths = length.(outrows)
    max_colwidths = [maximum(view(colwidths, :, i)) for i in 1:nc]
    totwidth = sum(max_colwidths) + 2 * (nc - 1)

    println(io, "F-test: $N models fitted on $(ftr.nobs) observations")
    println(io, '─'^totwidth)

    for r in 1:(nr + 1)
        for c in 1:nc
            cur_cell = outrows[r, c]
            cur_cell_len = length(cur_cell)

            padding = " "^(max_colwidths[c] - cur_cell_len)
            if c > 1
                padding = "  " * padding
            end

            print(io, padding)
            print(io, cur_cell)
        end
        print(io, "\n")
        r == 1 && println(io, '─'^totwidth)
    end
    return print(io, '─'^totwidth)
end

function ftest(r::LinearModel{T,<:ProbabilityWeights}) where {T}
    throw(ArgumentError("`ftest` for probability weighted models is not currently supported."))
end
