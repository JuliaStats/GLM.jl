module Glm

load("Distributions")
load("DataFrames")
using DataFrames, Distributions

import Base.\, Base.size
import Distributions.deviance, Distributions.mueta, Distributions.var

export                                  # types
    DGlmResp,                           # distributed GlmResp
    DensePred,
    DensePredQR,
    DensePredChol,
    DistPred,
    GlmResp,
    LinPred,
                                        # functions
    contr_treatment,# treatment contrasts
    drsum,          # sum of squared deviance residuals
    gl,             # generate levels
    glm,            # general interface
    glmfit,         # underlying workhorse
    indicators,     # generate dense or sparse indicator matrices
    linpred,        # linear predictor
    sqrtwrkwt,      # square root of the working weights
    updatebeta,
    updatemu,
    wrkresid,       # working residuals
    wrkresp,        # working response
    xtab,           # cross-tabulation
    xtabs           # another cross-tabulation

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
        if !(numel(eta) == numel(mu) == numel(offset) == numel(wts) == numel(y))
            error("mismatched sizes")
        end
        insupport(d, y)? new(d,l,eta,mu,offset,wts,y): error("elements of y not in distribution support")
    end
end

## outer constructor - the most common way of creating the object
function GlmResp(d::Distribution, l::Link, y::Vector{Float64})
    sz = size(y)
    wt = ones(Float64, sz)
    mu = mustart(d, y, wt)
    GlmResp(d, l, linkfun(l, mu), mu, zeros(Float64, sz), wt, y)
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

function updatemu{T<:Real}(r::GlmResp, linPr::AbstractArray{T})
    promote_shape(size(linPr), size(r.eta)) # size check
    for i=1:numel(linPr)
        r.eta[i] = linPr[i] + r.offset[i]
        r.mu[i]  = linkinv(r.l, r.eta[i])
    end
    deviance(r)
end
    
type DGlmResp                    # distributed response in a glm model
    d::Distribution
    l::Link
    eta::DArray{Float64,1,1}     # linear predictor
    mu::DArray{Float64,1,1}      # mean response
    offset::DArray{Float64,1,1}  # offset added to linear predictor (usually 0)
    wts::DArray{Float64,1,1}     # prior weights
    y::DArray{Float64,1,1}       # response
    ## FIXME: Add compatibility checks here
end

function DGlmResp(d::Distribution, l::Link, y::DArray{Float64,1,1})
    wt     = darray((T,d,da)->ones(T,d), Float64, size(y), distdim(y), y.pmap)
    offset = darray((T,d,da)->zeros(T,d), Float64, size(y), distdim(y), y.pmap)
    mu     = similar(y)
    @sync begin
        for p = y.pmap
            @spawnat p copy_to(localize(mu), d.mustart(localize(y), localize(wt)))
        end
    end
    dGlmResp(d, l, map_vectorized(link.linkFun, mu), mu, offset, wt, y)
end

DGlmResp(d::Distribution, y::DArray{Float64,1,1}) = DGlmResp(d, canonicallink(d), y)

## deviance( r::GlmResp) = deviance(r.d, r.mu, r.y, r.wts)
## devResid( r::GlmResp) = devResid(r.d, r.y, r.mu, r.wts)
## drsum(    r::GlmResp) = sum(devResid(r))
## mueta(    r::GlmResp) = mueta(r.l, r.eta)
## sqrtwrkwt(r::GlmResp) = mueta(r) .* sqrt(r.wts ./ var(r))
## var(      r::GlmResp) = var(r.d, r.mu)
## wrkresid( r::GlmResp) = (r.y - r.mu) ./ mueta(r)
## wrkresp(  r::GlmResp) = (r.eta - r.offset) + wrkresid(r)

abstract LinPred                        # linear predictor for statistical models
abstract DensePred <: LinPred           # linear predictor with dense X

                                        # linear predictor vector
linpred(p::LinPred, f::Real) = p.X * (p.beta0 + f * p.delbeta)
linpred(p::LinPred) = linpred(p, 1.)

type DensePredQR <: DensePred
    X::Matrix{Float64}                  # model matrix
    beta0::Vector{Float64}              # base coefficient vector
    delbeta::Vector{Float64}            # coefficient increment
    qr::QRDense{Float64}
    function DensePredQR(X::Matrix{Float64}, beta0::Vector{Float64})
        n, p = size(X)
        if length(beta0) != p error("dimension mismatch") end
        new(X, beta0, zeros(Float64, size(beta0)), qrd(X))
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
        new(X, beta0, zeros(Float64, size(beta0)), chold(X'X))
    end
end

## outer constructors
DensePredQR(X::Matrix{Float64}) = DensePredQR(X, zeros(Float64,(size(X,2),)))
DensePredQR{T<:Real}(X::Matrix{T}) = DensePredQR(float64(X))
DensePredChol(X::Matrix{Float64}) = DensePredChol(X, zeros(Float64,(size(X,2),)))
DensePredChol{T<:Real}(X::Matrix{T}) = DensePredChol(float64(X))

function delbeta(p::DensePredQR, y::Vector{Float64}, sqrtwt::Vector{Float64})
    p.qr.hh[:] = diagmm(sqrtwt, p.X)
    p.qr.tau[:] = LAPACK.geqrf!(p.qr.hh)[2]
    p.delbeta[:] = p.qr \ (sqrtwt .* y)
end

function delbeta(p::DensePredChol, y::Vector{Float64}, sqrtwt::Vector{Float64})
    WX = diagmm(sqrtwt, p.X)
    fac = LAPACK.potrf!('U', BLAS.syrk('U', 'T', 1.0, WX))
    if fac[2] != 0
        error("Singularity detected at column $(fac[2]) of weighted model matrix")
    end
    p.chol.LR[:] = fac[1]
    p.delbeta[:] = p.chol \ (WX'*(sqrtwt .* y))
end

At_mult_B{T <: Real}(A::DArray{T, 2, 1}, B::DArray{T, 2, 1}) = Ac_mult_B(A, B)

function Ac_mult_B{T <: Real}(A::DArray{T, 2, 1}, B::DArray{T, 2, 1})
    if (all(procs(A) != procs(B)) || all(dist(A) != dist(B)))
        # FIXME: B should be redistributed to match A
        error("Arrays A and B must be distributed similarly")
    end
    if is(A, B)
        return mapreduce(+, fetch, {@spawnat p BLAS.syrk('T', localize(A)) for p in procs(A)})
    end
    mapreduce(+, fetch, {@spawnat p Ac_mult_B(localize(A), localize(B)) for p in procs(A)})
end

function Ac_mult_B{T <: Real}(A::DArray{T, 2, 1}, B::DArray{T, 1, 1})
    if (all(procs(A) != procs(B)) || all(dist(A) != dist(B)))
        # FIXME: B should be redistributed to match A
        error("Arrays A and B must be distributed similarly")
    end
    mapreduce(+, fetch, {@spawnat p Ac_mult_B(localize(A), localize(B)) for p in procs(A)})
end

type DistPred{T} <: LinPred   # predictor with distributed (on rows) X
    X::DArray{T, 2, 1}        # model matrix
    beta::Vector{T}           # coefficient vector
    r::CholeskyDense{T}
    function DistPred(X, beta)
        if size(X, 2) != length(beta) error("dimension mismatch") end
        new(X, beta, chold(X'X))
    end
end

function (\)(A::DArray{Float64,2,1}, B::DArray{Float64,1,1})
    R   = Cholesky(A'A)
    LAPACK.potrs!('U', R, A'B)
end

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
    if !cvg
        error("failure to converge in $maxIter iterations")
    end
end

glmfit(p::DensePred, r::GlmResp) = glmfit(p, r, uint(30), 0.001, 1.e-6)

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
          
function glm(f::Formula, df::DataFrame, d::Distribution, l::Link, m::CompositeKind)
    if !(m <: LinPred) error("Composite type $m does not extend LinPred") end
    mf = model_frame(f, df)
    mm = model_matrix(mf)
    rr = GlmResp(d, l, vec(mm.response))
    dp = m(mm.model)
    GlmMod(mf, mm, rr, dp)
end
 
glm(f::Formula, df::DataFrame, d::Distribution, l::Link) = glm(f, df, d, l, DensePredQR)
    
glm(f::Formula, df::DataFrame, d::Distribution) = glm(f, df, d, canonicallink(d))
    
glm(f::Expr, df::DataFrame, d::Distribution) = glm(Formula(f), df, d)

# Generate levels - see the R documentation for gl
function gl(n::Integer, k::Integer, l::Integer)
    nk = n * k
    if l % nk != 0 error("length out must be a multiple of n * k") end
    aa = Array(Int, l)
    for j = 0:(l/nk - 1), i = 1:n
        aa[j * nk + (i - 1) * k + (1:k)] = i
    end
    PooledDataVector(aa)
end

gl(n::Integer, k::Integer) = gl(n, k, n*k)

# A cross-tabulation type.  Probably not a good design.
# Actually, this is just a one-way table
type xtab{T}
    vals::Array{T}
    counts::Array{Int, 1}
end

function xtab{T}(x::AbstractArray{T})
    d = Dict{T, Int}()
    for el in x d[el] = has(d, el) ? d[el] + 1 : 1 end
    kk = sort(keys(d))
    cc = Array(Int, numel(kk))
    for i in 1:numel(kk) cc[i] = d[kk[i]] end
    xtab(kk, cc)
end

# Another cross-tabulation function, this one leaves the result as a Dict
# Again, this is currently just for one-way tables.
function xtabs{T}(x::AbstractArray{T})
    d = Dict{T, length(x) > typemax(Int32) ? Int : Int32}()
    for el in x d[el] = has(d, el) ? d[el] + 1 : 1 end
    d
end

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

end # module
