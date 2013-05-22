type GlmResp <: ModResp               # response in a glm model
    y::Vector{Float64}                # response
    d::Distribution
    l::Link
    eta::Vector{Float64}              # linear predictor
    mu::Vector{Float64}               # mean response
    offset::Vector{Float64}           # offset added to linear predictor (usually 0)
    wts::Vector{Float64}              # prior weights
    function GlmResp(y::Vector{Float64}, d::Distribution, l::Link, 
                     eta::Vector{Float64}, mu::Vector{Float64}, offset::Vector{Float64},
                     wts::Vector{Float64})
        if !(length(eta) == length(mu) == length(offset) == length(wts) == length(y))
            error("mismatched sizes")
        end
        insupport(d, y)? new(y,d,l,eta,mu,offset,wts):
        error("elements of y not in distribution support")
    end
end

function GlmResp(y::Vector{Float64}, d::Distribution, l::Link)
    n  = length(y)
    wt = ones(n)
    mu = mustart(d, y, wt)
    GlmResp(y, d, l, linkfun(l, mu), mu, zeros(n), wt)
end

GlmResp(y::Vector{Float64}, d::Distribution) = GlmResp(y, d, canonicallink(d))
GlmResp{T<:Real}(y::Vector{T}, d::Distribution, args...) = GlmResp(float64(y), d, args...)

deviance( r::GlmResp) = deviance(r.d, r.mu, r.y, r.wts)
devresid( r::GlmResp) = devresid(r.d, r.y, r.mu, r.wts)
drsum(    r::GlmResp) = sum(devresid(r))
mueta(    r::GlmResp) = mueta(r.l, r.eta)
sqrtwrkwt(r::GlmResp) = mueta(r) .* sqrt(r.wts ./ var(r))
var(      r::GlmResp) = var(r.d, r.mu)
wrkresid( r::GlmResp) = (r.y - r.mu) ./ mueta(r)
wrkresp(  r::GlmResp) = (r.eta - r.offset) + wrkresid(r)

function updatemu(r::GlmResp, linPr::Vector{Float64})
    n = length(linPr)
    if length(r.mu) != n throw(LinAlg.LAPACK.DimensionMismatch("linPr")) end
    r.eta[:] = linPr
    if length(r.offset) == n r.eta += r.offset end
    r.mu = linkinv(r.l, r.eta)
    deviance(r)
end
updatemu(r::GlmResp, linPr) = updatemu(r, float64(vec(linPr)))

type GlmMod <: LinPredModel
    fr::ModelFrame
    mm::ModelMatrix
    rr::GlmResp
    pp::LinPred
    ff::Formula
    fit::Bool
end
scalepar(x::GlmMod) = 1. # generalize this - only appropriate for Bernoulli and Poisson

## Change this to use optional arguments for the form of the predictor?
function glm(f::Formula, df::AbstractDataFrame, d::Distribution, l::Link, m::DataType)
    if !(m <: LinPred) error("Composite type $m does not extend LinPred") end
    mf = ModelFrame(f, df)
    mm = ModelMatrix(mf)
    rr = GlmResp(dv(model_response(mf)), d, l)
    dp = m(mm.m)
    GlmMod(mf, mm, rr, dp, f, false)
end

glm(f::Formula, df::AbstractDataFrame, d::Distribution, l::Link) = glm(f, df, d, l, DensePredQR)
glm(f::Formula, df::AbstractDataFrame, d::Distribution) = glm(f, df, d, canonicallink(d))
glm(f::Expr, df::AbstractDataFrame, d::Distribution) = glm(Formula(f), df, d)
glm(f::String, df::AbstractDataFrame, d::Distribution) = glm(Formula(parse(f)[1]), df, d)

function fit(m::GlmMod; verbose=false, maxIter=30, minStepFac=0.001, convTol=1.e-6)
    if !m.fit 
        if maxIter < 1 error("maxIter must be positive") end
        if !(0 < minStepFac < 1) error("minStepFac must be in (0, 1)") end

        cvg = false
        devold = Inf
        p = m.pp
        r = m.rr
        for i=1:maxIter
            delbeta(p, wrkresp(r), sqrtwrkwt(r))
            dev  = updatemu(r, linpred(p))
            crit = (devold - dev)/dev
            if verbose println("$i: $dev, $crit") end
            if abs(crit) < convTol
                cvg = true
                break
            end
            if (dev >= devold)
                error("code needed to handle the step-factor case")
            end
            devold = dev
        end
        if !cvg error("failure to converge in $maxIter iterations") end
        installbeta(p)
        m.fit = true
    end
    m
end

function coeftable(mm::GlmMod)
    cc = coef(mm)
    se = stderr(mm)
    zz = cc ./ se
    DataFrame({cc, se, zz, 2.0 * ccdf(Normal(), abs(zz))},
              ["Estimate","Std.Error","z value", "Pr(>|z|)"])
end

function confint(obj::GlmMod, level::Real)
    cft = coeftable(obj)
    hcat(coef(obj),coef(obj)) + cft["Std.Error"] *
    quantile(Normal(), (1. - level)/2.) * [1. -1.]
end
confint(obj::LmMod) = confint(obj, 0.95)
