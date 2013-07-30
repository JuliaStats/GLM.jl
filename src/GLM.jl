using DataFrames, Distributions, NumericExtensions

module GLM

using DataFrames, Distributions, NumericExtensions
using Base.LinAlg.LAPACK: geqrt3!, potrf!, potri!, potrs!
using Base.LinAlg.BLAS: syrk!, gemv!, gemm!, symmetrize!

import Base: (\), scale, size, show
import Distributions: fit, logpdf
import DataFrames: ModelFrame, ModelMatrix, model_response
import NumericExtensions: evaluate, result_type # to be able to define functors

export                                  # types
    CauchitLink,
    CloglogLink,
    DensePred,
    DensePredQR,
    DensePredChol,
    GlmMod,
    GlmResp,
    IdentityLink,
    InverseLink,
    Link,
    LinPred,
    LinPredModel,
    LogitLink,
    LogLink,
    LmMod,
    LmResp,
    ProbitLink,
                                        # functions
    canonicallink,  # canonical link function for a distribution
    coef,           # estimated coefficients
    coeftable,      # coefficients, standard errors, etc.
    confint,        # confidence intervals on coefficients
    contr_treatment,# treatment contrasts
    delbeta!,       # evaluate the increment in the coefficient vector
    deviance,       # deviance of fitted and observed responses
    devresid,       # vector of squared deviance residuals
    df_residual,    # degrees of freedom for residuals
    drsum,          # sum of squared deviance residuals
    formula,        # extract the formula from a model
    glm,            # general interface
    linkfun,        # link function mapping mu to eta, the linear predictor
    linkfun!,       # mutating link function
    linkinv,        # inverse link mapping eta to mu
    linkinv!,       # mutating inverse link
    linpred,        # linear predictor
    lm,             # linear model (QR factorization)
    lmc,            # linear model (Cholesky factorization)          
    mueta,          # derivative of inverse link
    mueta!,         # mutating derivative of inverse link
    mustart,        # derive starting values for the mu vector
    nobs,           # total number of observations
    predict,        # make predictions
    residuals,      # extractor for residuals
    sqrtwrkwt,      # square root of the working weights
    stderr,         # standard errors of the coefficients
    updatemu!,      # mutating update of the response type from the linear predictor
    var!,           # mutating variance function
    vcov,           # estimated variance-covariance matrix of coef
    wrkresid,       # extract the working residuals              
    wrkresid!,      # mutating working residuals function
    wrkresp         # working response

typealias FP FloatingPoint

abstract ModResp                        # model response

abstract LinPred             # linear predictor for statistical models
abstract DensePred <: LinPred          # linear predictor with dense X

## Return the linear predictor vector
linpred(p::LinPred, f::Real=1.) = p.X.m * (f == 0. ? p.beta0 : fma(p.beta0, p.delbeta, f))

## Install beta0 + f*delbeta as beta0 and zero out delbeta
function installbeta!(p::LinPred, f::Real=1.)
    fma!(p.beta0, p.delbeta, f)
    fill!(p.delbeta, 0.)
    p.beta0
end

typealias BlasReal Union(Float32,Float64)
    
type DensePredQR{T<:BlasReal} <: DensePred
    X::ModelMatrix{T}             # model matrix
    beta0::Vector{T}              # base coefficient vector
    delbeta::Vector{T}            # coefficient increment
    qr::QR{T}
    function DensePredQR(X::ModelMatrix{T}, beta0::Vector{T})
        n,p = size(X); length(beta0) == p || error("dimension mismatch")
        new(X, beta0, zeros(T,p), qrfact(X.m))
    end
end
DensePredQR{T<:BlasReal}(X::ModelMatrix{T}) = DensePredQR{T}(X, zeros(T,size(X,2)))

delbeta!{T<:BlasReal}(p::DensePredQR{T}, r::Vector{T}) = (p.delbeta = p.qr\r; p)
              
type DensePredChol{T<:BlasReal} <: DensePred
    X::ModelMatrix{T}                   # model matrix
    beta0::Vector{T}                    # base vector for coefficients
    delbeta::Vector{T}                  # coefficient increment
    chol::Cholesky{T}
    function DensePredChol(X::ModelMatrix{T}, beta0::Vector{T})
        n,p = size(X); length(beta0) == p || error("dimension mismatch")
        new(X, beta0, zeros(T,p), cholfact(X.m'X.m))
    end
end
DensePredChol{T<:BlasReal}(X::ModelMatrix{T}) = DensePredChol{T}(X, zeros(T,size(X,2)))

solve!{T<:BlasReal}(C::Cholesky{T}, B::StridedVecOrMat{T}) = potrs!(C.uplo, C.UL, B)

function delbeta!{T<:BlasReal}(p::DensePredChol{T}, r::Vector{T})
    solve!(p.chol, gemv!('T', 1.0, p.X.m, r, 0.0, p.delbeta))
    p
end

function delbeta!{T<:BlasReal}(p::DensePredChol{T}, r::Vector{T}, wt::Vector{T}, scr::Matrix{T})
    vbroadcast!(Multiply(), scr, p.X.m, wt, 1)
    fac, info = potrf!('U', gemm!('T', 'N', 1.0, scr, p.X.m, 0.0, p.chol.UL))
    info == 0 || error("Singularity detected at column $info of weighted model matrix")
    solve!(p.chol, gemv!('T', 1.0, scr, r, 0.0, p.delbeta))
    p
end

abstract LinPredModel  # statistical model based on a linear predictor

coef(x::LinPred) = x.beta0
coef(x::LinPredModel) = coef(x.pp)

df_residual(x::LinPredModel) = df_residual(x.pp)
df_residual(x::DensePred) = size(x.X, 1) - length(x.beta0)
    
vcov(x::LinPredModel) = scale(x,true) * vcov(x.pp)
vcov(x::DensePredChol) = inv(x.chol)
vcov(x::DensePredQR) = symmetrize!(potri!('U', x.qr[:R])[1])

stderr(x::LinPredModel) = sqrt(diag(vcov(x)))

function show(io::IO, obj::LinPredModel)
    println(io, "\n$(obj.ff)\n\nCoefficients:\n")
    println(io, coeftable(obj))
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

ModelFrame(obj::LinPredModel) = obj.fr
ModelMatrix(obj::LinPredModel) = obj.pp.X
model_response(obj::LinPredModel) = obj.rr.y

fitted(m::LinPredModel) = m.rr.mu
predict(mm::LinPredModel) = fitted(mm)
formula(obj::LinPredModel) = ModelFrame(obj).formula
nobs(obj::LinPredModel) = length(model_response(obj))
residuals(obj::LinPredModel) = residuals(obj.rr)

include("lm.jl")
include("glmtools.jl")
include("glmfit.jl")

end # module
