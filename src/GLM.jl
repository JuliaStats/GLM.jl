using DataFrames, Distributions

module GLM

using DataFrames, Distributions         # This seems to be necessary within the module

import Base: (\), scale, size, show
import Distributions: deviance, devresid, fit, mueta, var
import DataFrames: model_frame, model_matrix, model_response

export                                  # types
    DensePred,
    DensePredQR,
    DensePredChol,
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
    df_residual,    # degrees of freedom for residuals
    drsum,          # sum of squared deviance residuals
    family,
    formula,
    glm,            # general interface
    indicators,     # generate dense or sparse indicator matrices
    linpred,        # linear predictor
    lm,             # linear model
    nobs,           # total number of observations
    predict,        # make predictions
    residuals,      # extractor for residuals
    scalepar,       # estimate of scale parameter (sigma^2 for linear models)
    sqrtwrkwt,      # square root of the working weights
    stderr,         # standard errors of the coefficients
    vcov,           # estimated variance-covariance matrix of coef
    wrkresid,       # working residuals
    wrkresp         # working response

abstract ModResp                        # model response

abstract LinPred             # linear predictor for statistical models
abstract DensePred <: LinPred          # linear predictor with dense X

## Return the linear predictor vector
linpred(p::LinPred, f::Real) = p.X * (p.beta0 + f * p.delbeta)
linpred(p::LinPred) = linpred(p, 1.0)

## Install beta0 + f*delbeta as beta0 and zero out delbeta
function installbeta(p::LinPred, f::Real)
    p.beta0 += f * p.delbeta
    p.delbeta[:] = zeros(length(p.delbeta))
    p.beta0
end
installbeta(p::LinPred) = installbeta(p, 1.0)

type DensePredQR <: DensePred
    X::Matrix{Float64}                  # model matrix
    beta0::Vector{Float64}              # base coefficient vector
    delbeta::Vector{Float64}            # coefficient increment
    qr::QR{Float64}
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
    chol::Cholesky{Float64}
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
    p.qr.vs[:] = scale(sqrtwt, p.X)
    p.qr.T[:] = LinAlg.LAPACK.geqrt3!(p.qr.vs)[2]
    p.delbeta[:] = p.qr \ (sqrtwt .* r)
end

function delbeta(p::DensePredChol, r::Vector{Float64}, sqrtwt::Vector{Float64})
    WX = scale(sqrtwt, p.X)
    if LinAlg.LAPACK.potrf!('U', LinAlg.BLAS.syrk!('U', 'T', 1.0, WX, 0.0, p.chol.LR))[2] != 0
        error("Singularity detected at column $(fac[2]) of weighted model matrix")
    end
    p.delbeta[:] = p.chol \ (WX'*(sqrtwt .* r))
end

delbeta(p::DensePred, r::Vector{Float64}) = delbeta(p, r, ones(length(r)))

abstract LinPredModel  # statistical model based on a linear predictor

dv(da::DataArray) = da.data # return the data values - move this to DataFrames?
dv(pda::PooledDataArray) = pda.refs
dv{T<:Number}(vv::Vector{T}) = vv

if false
    ## Probably no longer necessary
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
end

coef(x::LinPred) = x.beta0
coef(x::LinPredModel) = coef(fit(x).pp)

deviance(x::LinPredModel) = deviance(fit(x).rr)
df_residual(x::LinPredModel) = df_residual(x.pp)
df_residual(x::DensePred) = size(x.X, 1) - length(x.beta0)
    
vcov(x::LinPredModel) = scalepar(fit(x)) * vcov(x.pp)
vcov(x::DensePredChol) = inv(x.chol)
vcov(x::DensePredQR) = LinAlg.BLAS.symmetrize!(LinAlg.LAPACK.potri!('U', x.qr[:R])[1])

stderr(x::LinPredModel) = sqrt(diag(vcov(x)))

function show(io::IO, obj::LinPredModel)
    @printf("\n%s\n\nCoefficients:\n", obj.ff)
    println(io, coeftable(obj))
#    @printf("R-squared: %0.4f\n", 0.0) # TODO: obj.r_squared)
end

## function show(io::IO, obj::GlmMod)
##     cc = coef(obj)
##     se = stderr(obj)
##     zz = cc ./ se
##     pp = 2.0 * ccdf(Normal(), abs(zz))
##     @printf("\n%s\n\nCoefficients:\n", obj.fr.formula)
##     @printf("         Term    Estimate  Std. Error     t value    Pr(>|t|)\n")
##     N = length(cc)
##     for i = 1:N
##         @printf(" %12s%12.5f%12.5f%12.3f%12.3f %-3s\n",
##                 obj.mm.model_colnames[i],
##                 cc[i],
##                 se[i],
##                 zz[i],
##                 pp[i],
##                 p_value_stars(pp[i]))
##     end
##     println("---\nSignif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1\n")
##     @printf("R-squared: %0.4f\n", 0.0) # TODO: obj.r_squared)
## end

## function p_value_stars(p_value::Float64)
##     if p_value < 0.001
##         return "***"
##     elseif p_value < 0.01
##         return "**"
##     elseif p_value < 0.05
##         return "*"
##     elseif p_value < 0.1
##         return "."
##     else
##         return " "
##     end
## end

model_frame(obj::LinPredModel) = obj.fr
model_matrix(obj::LinPredModel) = obj.mm
model_response(obj::LinPredModel) = obj.rr.y

fitted(m::LinPredModel) = m.rr.mu
predict(mm::LinPredModel) = fitted(mm)
formula(obj::LinPredModel) = model_frame(obj).formula
nobs(obj::LinPredModel) = length(model_response(obj))
residuals(obj::LinPredModel) = residuals(obj.rr)

include("lm.jl")
include("glmfit.jl")

end # module
