
                  
## Return the linear predictor vector
linpred(p::LinPred, f::Real=1.) = p.X * (f == 0. ? p.beta0 : fma(p.beta0, p.delbeta, f))

## Install beta0 + f*delbeta as beta0 and zero out delbeta
function installbeta!(p::LinPred, f::Real=1.)
    fma!(p.beta0, p.delbeta, f)
    fill!(p.delbeta, 0.)
    p.beta0
end

typealias BlasReal Union(Float32,Float64)
    
type DensePredQR{T<:BlasReal} <: DensePred
    X::Matrix{T}                  # model matrix
    beta0::Vector{T}              # base coefficient vector
    delbeta::Vector{T}            # coefficient increment
    qr::QRCompactWY{T}
    function DensePredQR(X::Matrix{T}, beta0::Vector{T})
        n,p = size(X); length(beta0) == p || error("dimension mismatch")
        new(X, beta0, zeros(T,p), qrfact(X))
    end
end
DensePredQR{T<:BlasReal}(X::Matrix{T}) = DensePredQR{T}(X, zeros(T,size(X,2)))

cholfact{T<:FP}(p::DensePredQR{T}) = Cholesky{T}(p.qr[:R],'U')

delbeta!{T<:BlasReal}(p::DensePredQR{T}, r::Vector{T}) = (p.delbeta = p.qr\r; p)
              
type DensePredChol{T<:BlasReal} <: DensePred
    X::Matrix{T}                   # model matrix
    beta0::Vector{T}               # base vector for coefficients
    delbeta::Vector{T}             # coefficient increment
    chol::Cholesky{T}
    function DensePredChol(X::Matrix{T}, beta0::Vector{T})
        n,p = size(X); length(beta0) == p || error("dimension mismatch")
        new(X, beta0, zeros(T,p), cholfact(X'X))
    end
end
DensePredChol{T<:BlasReal}(X::Matrix{T}) = DensePredChol{T}(X, zeros(T,size(X,2)))

solve!{T<:BlasReal}(C::Cholesky{T}, B::StridedVecOrMat{T}) = potrs!(C.uplo, C.UL, B)

cholfact{T<:FP}(p::DensePredChol{T}) = (c = p.chol; Cholesky{T}(copy(c.UL),c.uplo))

function delbeta!{T<:BlasReal}(p::DensePredChol{T}, r::Vector{T})
    solve!(p.chol, gemv!('T', 1.0, p.X, r, 0.0, p.delbeta))
    p
end

function delbeta!{T<:BlasReal}(p::DensePredChol{T}, r::Vector{T}, wt::Vector{T}, scr::Matrix{T})
    vbroadcast!(Multiply(), scr, p.X, wt, 1)
    fac, info = potrf!('U', gemm!('T', 'N', 1.0, scr, p.X, 0.0, p.chol.UL))
    info == 0 || error("Singularity detected at column $info of weighted model matrix")
    solve!(p.chol, gemv!('T', 1.0, scr, r, 0.0, p.delbeta))
    p
end


coef(x::LinPred) = x.beta0
coef(x::LinPredModel) = coef(x.pp)

df_residual(x::LinPredModel) = df_residual(x.pp)
df_residual(x::DensePred) = size(x.X, 1) - length(x.beta0)
    
vcov(x::LinPredModel) = scale(x,true) * inv(cholfact(x.pp))
#vcov(x::DensePredChol) = inv(x.chol)
#vcov(x::DensePredQR) = copytri!(potri!('U', x.qr[:R]), 'U')

cor(x::LinPredModel) = (invstd = map(RcpFun(),stderr(x)); scale!(invstd,scale!(vcov(x),invstd)))

stderr(x::LinPredModel) = sqrt(diag(vcov(x)))

function show(io::IO, obj::LinPredModel)
    println(io, isdefined(obj, :ff) ? obj.ff : "Explicitly specified design matrix",
            "\n\nCoefficients:", coeftable(obj))
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
