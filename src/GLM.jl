module GLM

using DataFrames, Distributions

import Base.\, Base.size, Base.show
import Distributions.deviance, Distributions.mueta, Distributions.var
import DataFrames.model_frame, DataFrames.model_matrix

export                                  # types
#    DGlmResp,
    DensePred,
    DensePredQR,
    DensePredChol,
#    DistPred,
    GlmMod,
    GlmResp,
    LinPred,
    LinPredModel,
    LmMod,
    LmResp,
                                        # functions
    coef,           # estimated coefficients
    coeftable,      # coefficients, standard errors, etc.
    confint,        # confidence intervals on coefficients
    contr_treatment,# treatment contrasts
#    delbeta,        # an internal function for calculating the beta increment
    deviance,       # deviance of GLM
    df_residual,    # degrees of freedom for residuals
    drsum,          # sum of squared deviance residuals
    family,
    formula,
    glm,            # general interface
    glmfit,         # underlying workhorse
    indicators,     # generate dense or sparse indicator matrices
#   installbeta,     # an internal function for installing a new beta0
    linpred,        # linear predictor
    lm,             # linear model
    lmfit,          # linear model
    nobs,           # total number of observations
    predict,        # make predictions
    residuals,      # extractor for residuals
    scale,          # estimate of scale parameter (sigma^2 for linear models)
    sqrtwrkwt,      # square root of the working weights
    stderr,         # standard errors of the coefficients
    updatemu,
    vcov,           # estimated variance-covariance matrix of coef
    wrkresid,       # working residuals
    wrkresp         # working response

abstract ModResp                      # model response

type GlmResp <: ModResp               # response in a glm model
    d::Distribution                  
    l::Link
    eta::Vector{Float64}              # linear predictor
    mu::Vector{Float64}               # mean response
    offset::Vector{Float64}           # offset added to linear predictor (usually 0)
    wts::Vector{Float64}              # prior weights
    y::Vector{Float64}                # response
    function GlmResp(d::Distribution, l::Link, eta::Vector{Float64},
                     mu::Vector{Float64}, offset::Vector{Float64},
                     wts::Vector{Float64},y::Vector{Float64})
        if !(length(eta) == length(mu) == length(offset) == length(wts) == length(y))
            error("mismatched sizes")
        end
        insupport(d, y)? new(d,l,eta,mu,offset,wts,y): error("elements of y not in distribution support")
    end
end

## outer constructor - the most common way of creating the object
function GlmResp(d::Distribution, l::Link, y::Vector{Float64})
    n  = length(y)
    wt = ones(n)
    mu = mustart(d, y, wt)
    GlmResp(d, l, linkfun(l, mu), mu, zeros(n), wt, y)
end

GlmResp(d::Distribution, y::Vector{Float64}) = GlmResp(d, canonicallink(d), y)

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
    if length(r.mu) != n throw(LAPACK.LapackDimMisMatch("linPr")) end
    r.eta[:] = linPr
    if length(r.offset) == n r.eta += r.offset end
    r.mu = linkinv(r.l, r.eta)
    deviance(r)
end

updatemu(r::GlmResp, linPr) = updatemu(r, float64(vec(linPr)))

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
    if length(r.mu) != n throw(LAPACK.LapackDimMisMatch("linpr")) end
    r.mu[:] = linPr
    if (length(r.offset) == n) r.mu += r.offset end
    deviance(r)
end

updatemu(r::LmResp, linPr) = updatemu(r, float64(vec(linPr)))

wrkresid(r::LmResp) = (r.y - r.mu) .* sqrt(r.wts)
drsum(r::LmResp)    = sum(wrkresid(r) .^ 2)
deviance(r::LmResp) = drsum(r)
residuals(r::LmResp)= wrkresid(r)
                          
## type DGlmResp                    # distributed response in a glm model
##     d::Distribution
##     l::Link
##     eta::DArray{Float64,1,1}     # linear predictor
##     mu::DArray{Float64,1,1}      # mean response
##     offset::DArray{Float64,1,1}  # offset added to linear predictor (usually 0)
##     wts::DArray{Float64,1,1}     # prior weights
##     y::DArray{Float64,1,1}       # response
##     ## FIXME: Add compatibility checks here
## end

## function DGlmResp(d::Distribution, l::Link, y::DArray{Float64,1,1})
##     wt     = darray((T,d,da)->ones(T,d), Float64, size(y), distdim(y), y.pmap)
##     offset = darray((T,d,da)->zeros(T,d), Float64, size(y), distdim(y), y.pmap)
##     mu     = similar(y)
##     @sync begin
##         for p = y.pmap
##             @spawnat p copy_to(localize(mu), d.mustart(localize(y), localize(wt)))
##         end
##     end
##     dGlmResp(d, l, map_vectorized(link.linkFun, mu), mu, offset, wt, y)
## end

## DGlmResp(d::Distribution, y::DArray{Float64,1,1}) = DGlmResp(d, canonicallink(d), y)

abstract LinPred                        # linear predictor for statistical models
abstract DensePred <: LinPred           # linear predictor with dense X

                                        # linear predictor vector
linpred(p::LinPred, f::Real) = p.X * (p.beta0 + f * p.delbeta)
linpred(p::LinPred) = linpred(p, 1.0)
function installbeta(p::LinPred, f::Real)
    p.beta0 += f * p.delbeta
    p.delbeta[:] = zeros(length(p.delbeta))
end
installbeta(p::LinPred) = installbeta(p, 1.0)

type DensePredQR <: DensePred
    X::Matrix{Float64}                  # model matrix
    beta0::Vector{Float64}              # base coefficient vector
    delbeta::Vector{Float64}            # coefficient increment
    qr::QRDense{Float64}
    function DensePredQR(X::Matrix{Float64}, beta0::Vector{Float64})
        n, p = size(X)
        if length(beta0) != p error("dimension mismatch") end
        new(X, beta0, zeros(Float64, size(beta0)), qrfact(X))
    end
end

type DensePredChol <: DensePred
    X::Matrix{Float64}                  # model matrix
    beta0::Vector{Float64}              # base vector for coefficients
    delbeta::Vector{Float64}            # coefficient increment
    chol::CholeskyDense{Float64}
    function DensePredChol(X::Matrix{Float64}, beta0::Vector{Float64})
        n, p = size(X)
        if length(beta0) != p error("dimension mismatch") end
        new(X, beta0, zeros(Float64, size(beta0)), cholfact(X'X))
    end
end

## outer constructors
DensePredQR(X::Matrix{Float64}) = DensePredQR(X, zeros(Float64,(size(X,2),)))
DensePredQR{T<:Real}(X::Matrix{T}) = DensePredQR(float64(X))
DensePredChol(X::Matrix{Float64}) = DensePredChol(X, zeros(Float64,(size(X,2),)))
DensePredChol{T<:Real}(X::Matrix{T}) = DensePredChol(float64(X))

function delbeta(p::DensePredQR, r::Vector{Float64}, sqrtwt::Vector{Float64})
    p.qr.hh[:] = diagmm(sqrtwt, p.X)
    p.qr.tau[:] = LAPACK.geqrf!(p.qr.hh)[2]
    p.delbeta[:] = p.qr \ (sqrtwt .* r)
end

function delbeta(p::DensePredChol, r::Vector{Float64}, sqrtwt::Vector{Float64})
    WX = diagmm(sqrtwt, p.X)
    if LAPACK.potrf!('U', BLAS.syrk!('U', 'T', 1.0, WX, 0.0, p.chol.LR))[2] != 0
        error("Singularity detected at column $(fac[2]) of weighted model matrix")
    end
    p.delbeta[:] = p.chol \ (WX'*(sqrtwt .* r))
end

delbeta(p::DensePred, r::Vector{Float64}) = delbeta(p, r, ones(length(r)))

## At_mul_B{T <: Real}(A::DArray{T, 2, 1}, B::DArray{T, 2, 1}) = Ac_mul_B(A, B)

## function Ac_mul_B{T <: Real}(A::DArray{T, 2, 1}, B::DArray{T, 2, 1})
##     if (all(procs(A) != procs(B)) || all(dist(A) != dist(B)))
##         # FIXME: B should be redistributed to match A
##         error("Arrays A and B must be distributed similarly")
##     end
##     if is(A, B)
##         return mapreduce(+, fetch, {@spawnat p BLAS.syrk('T', localize(A)) for p in procs(A)})
##     end
##     mapreduce(+, fetch, {@spawnat p Ac_mul_B(localize(A), localize(B)) for p in procs(A)})
## end

## function Ac_mul_B{T <: Real}(A::DArray{T, 2, 1}, B::DArray{T, 1, 1})
##     if (all(procs(A) != procs(B)) || all(dist(A) != dist(B)))
##         # FIXME: B should be redistributed to match A
##         error("Arrays A and B must be distributed similarly")
##     end
##     mapreduce(+, fetch, {@spawnat p Ac_mul_B(localize(A), localize(B)) for p in procs(A)})
## end

## type DistPred{T} <: LinPred   # predictor with distributed (on rows) X
##     X::DArray{T, 2, 1}        # model matrix
##     beta::Vector{T}           # coefficient vector
##     r::CholeskyDense{T}
##     function DistPred(X, beta)
##         if size(X, 2) != length(beta) error("dimension mismatch") end
##         new(X, beta, chold(X'X))
##     end
## end

## function (\)(A::DArray{Float64,2,1}, B::DArray{Float64,1,1})
##     R   = Cholesky(A'A)
##     LAPACK.potrs!('U', R, A'B)
## end

function glmfit(p::DensePred, r::GlmResp, maxIter::Integer, minStepFac::Float64, convTol::Float64)
    if maxIter < 1 error("maxIter must be positive") end
    if !(0 < minStepFac < 1) error("minStepFac must be in (0, 1)") end

    cvg = false
    devold = Inf
    for i=1:maxIter
        delbeta(p, wrkresp(r), sqrtwrkwt(r))
        dev  = updatemu(r, linpred(p))
        crit = (devold - dev)/dev
        println("$i: $dev, $crit")
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
end

glmfit(p::DensePred, r::GlmResp) = glmfit(p, r, uint(30), 0.001, 1.e-6)

function lmfit(p::DensePred, r::LmResp)
    delbeta(p, wrkresid(r), sqrt(r.wts))
    updatemu(r, linpred(p))
    installbeta(p)
end

abstract LinPredModel  # statistical model based on a linear predictor

type GlmMod <: LinPredModel
    fr::ModelFrame
    mm::ModelMatrix
    rr::GlmResp
    pp::LinPred
    function GlmMod(fr, mm, rr, pp)
        glmfit(pp, rr)
        new(fr, mm, rr, pp)
    end
end
          
function glm(f::Formula, df::AbstractDataFrame, d::Distribution, l::Link, m::CompositeKind)
    if !(m <: LinPred) error("Composite type $m does not extend LinPred") end
    mf = model_frame(f, df)
    mm = model_matrix(mf)
    rr = GlmResp(d, l, vec(mm.response))
    dp = m(mm.model)
    GlmMod(mf, mm, rr, dp)
end
 
glm(f::Formula, df::AbstractDataFrame, d::Distribution, l::Link) = glm(f, df, d, l, DensePredQR)
    
glm(f::Formula, df::AbstractDataFrame, d::Distribution) = glm(f, df, d, canonicallink(d))
    
glm(f::Expr, df::AbstractDataFrame, d::Distribution) = glm(Formula(f), df, d)

glm(f::String, df::AbstractDataFrame, d::Distribution) = glm(Formula(parse(f)[1]), df, d)

type LmMod <: LinPredModel
    fr::ModelFrame
    mm::ModelMatrix
    rr::LmResp
    pp::LinPred
    function LmMod(fr, mm, rr, pp)
        lmfit(pp, rr)
        new(fr, mm, rr, pp)
    end
end
          
function lm(f::Formula, df::AbstractDataFrame, m::CompositeKind)
    if !(m <: LinPred) error("Composite type $m does not extend LinPred") end
    mf = model_frame(f, df)
    mm = model_matrix(mf)
    rr = LmResp(vec(mm.response))
    dp = m(mm.model)
    LmMod(mf, mm, rr, dp)
end
 
lm(f::Formula, df::AbstractDataFrame) = lm(f, df, DensePredQR)
    
lm(f::Expr, df::AbstractDataFrame) = lm(Formula(f), df)

lm(f::String, df::AbstractDataFrame) = lm(Formula(parse(f)[1]), df)

## dense or sparse matrix of indicators of the levels of a vector
function indicators{T}(x::AbstractVector{T}, sparseX::Bool)
    levs = sort!(unique(x))
    nx   = length(x)
    nlev = length(levs)
    d    = Dict{T, Int}()
    for i in 1:nlev d[levs[i]] = i end
    ii   = 1:nx
    jj   = [d[el] for el in x]
    if sparseX return sparse(int32(ii), int32(jj), 1.), levs end
    X    = zeros(nx, nlev)
    for i in ii X[i, jj[i]] = 1. end
    X, levs
end

## default is dense indicators
indicators{T}(x::AbstractVector{T}) = indicators(x, false)

function contr_treatment(n::Int, base::Int, contrasts::Bool, sparse::Bool)
    contr = sparse ? speye(n) : eye(n)
    if !contrasts return contr end
    if n < 2
        error(sprintf("contrasts not defined for %d degrees of freedom", n - 1))
    end
    contr[:, [1:(base-1), (base+1):n]]
end

contr_treatment(n::Int, base::Int, contrasts::Bool) = contr_treatment(n, base, contrasts, false)
contr_treatment(n::Int, base::Int) = contr_treatment(n, base, true, false)
contr_treatment(n::Int) = contr_treatment(n, 1, true, false)

coef(x::LinPred) = x.beta0
coef(x::LinPredModel) = coef(x.pp)

deviance(x::LinPredModel) = deviance(x.rr)
df_residual(x::LmMod) = df_residual(x.pp)
df_residual(x::DensePred) = size(x.X, 1) - length(x.beta0)
    
vcov(x::LinPredModel) = scale(x) * vcov(x.pp)
vcov(x::DensePredChol) = inv(x.chol)
vcov(x::DensePredQR) = BLAS.symmetrize!(LAPACK.potri!('U', x.qr.hh[1:length(x.beta0),:])[1])

scale(x::LmMod) = deviance(x)/df_residual(x)
scale(x::GlmMod) = 1.

stderr(x::LinPredModel) = sqrt(diag(vcov(x)))

function coeftable(mm::LmMod)
    cc = coef(mm)
    se = stderr(mm)
    tt = cc ./ se
    DataFrame({cc, se, tt, ccdf(FDist(1, df_residual(mm)), tt .* tt)},
              ["Estimate","Std.Error","t value", "Pr(>|t|)"])
end

function coeftable(mm::GlmMod)
    cc = coef(mm)
    se = stderr(mm)
    zz = cc ./ se
    DataFrame({cc, se, zz, 2.0 * ccdf(Normal(), abs(zz))},
              ["Estimate","Std.Error","z value", "Pr(>|z|)"])
end

include("show.jl")

predict(mm::LmMod) = mm.rr.mu

function predict(mm::LmMod, newx::Matrix)
    # TODO: Need to add intercept
    newx * coef(mm)
end

family(obj::LmMod) = {:family => "gaussian", :link => "identity"}
formula(obj::LmMod) = obj.fr.formula
model_frame(obj::LmMod) = obj.fr
model_matrix(obj::LmMod) = obj.mm
nobs(obj::LmMod) = size(obj.mm.model, 1)
residuals(obj::LmMod) = residuals(obj.rr)
function confint(obj::LmMod, level::Real)
    cft = coeftable(obj)
    hcat(coef(obj),coef(obj)) + cft["Std.Error"] *
    quantile(TDist(df_residual(obj)), (1. - level)/2.) * [1. -1.]
end
confint(obj::LmMod) = confint(obj, 0.95)

end # module
