## FIXME: allow wts and offset to have length 0
type LmResp <: ModResp                # response in a linear model
    mu::Vector{Float64}               # mean response
    offset::Vector{Float64}           # offset added to linear predictor (usually 0)
    wts::Vector{Float64}              # prior weights
    y::Vector{Float64}                # response
    function LmResp(mu::Vector{Float64}, offset::Vector{Float64},
                    wts::Vector{Float64},y::Vector{Float64})
        if !(length(mu) == length(offset) == length(wts) == length(y))
            error("mismatched sizes")
        end
        new(mu,offset,wts,y)
    end
end

function LmResp(y::Vector{Float64})
    n = length(y)
    LmResp(zeros(n), zeros(n), ones(n), y)
end

function updatemu(r::LmResp, linPr::Vector{Float64})
    n = length(linPr)
    if length(r.mu) != n throw(LinAlg.LAPACK.DimensionMismatch("linpr")) end
    r.mu[:] = linPr
    if (length(r.offset) == n) r.mu += r.offset end
    deviance(r)
end
updatemu(r::LmResp, linPr) = updatemu(r, float64(vec(linPr)))

wrkresid(r::LmResp) = (r.y - r.mu) .* sqrt(r.wts)
drsum(r::LmResp)    = sum(wrkresid(r) .^ 2)
deviance(r::LmResp) = drsum(r)
residuals(r::LmResp)= wrkresid(r)

type LmMod <: LinPredModel
    fr::ModelFrame
    mm::ModelMatrix
    rr::LmResp
    pp::LinPred
    ff::Formula
    fit::Bool                  # has the model been fit?
end

function lm(f::Formula, df::AbstractDataFrame, m::DataType)
    if !(m <: LinPred) error("Composite type $m does not extend LinPred") end
    mf = ModelFrame(f, df)
    mm = ModelMatrix(mf)
    rr = LmResp(dv(model_response(mf)))
    dp = m(mm.m)
    LmMod(mf, mm, rr, dp, f, false)
end
lm(f::Formula, df::AbstractDataFrame) = lm(f, df, DensePredQR)
lm(f::Expr, df::AbstractDataFrame) = lm(Formula(f), df)
lm(f::String, df::AbstractDataFrame) = lm(Formula(parse(f)[1]), df)

function fit(m::LmMod)
    p = m.pp
    r = m.rr
    if !m.fit
        delbeta(p, wrkresid(r), sqrt(r.wts))
        updatemu(r, linpred(p))
        installbeta(p)
        m.fit = true
    end
    m
end

scalepar(x::LmMod) = deviance(x)/df_residual(x)

function coeftable(mm::LmMod)
    cc = coef(mm)
    se = stderr(mm)
    tt = cc ./ se
    DataFrame({cc, se, tt, ccdf(FDist(1, df_residual(mm)), tt .* tt)},
              ["Estimate","Std.Error","t value", "Pr(>|t|)"])
end

function predict(mm::LmMod, newx::Matrix)
    # TODO: Need to add intercept
    newx * coef(fit(mm))
end

function confint(obj::LmMod, level::Real)
    cft = coeftable(obj)
    hcat(coef(obj),coef(obj)) + cft["Std.Error"] *
    quantile(TDist(df_residual(obj)), (1. - level)/2.) * [1. -1.]
end
confint(obj::LmMod) = confint(obj, 0.95)
