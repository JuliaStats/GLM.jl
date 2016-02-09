## Return the linear predictor vector
function linpred!(out, p::LinPred, f::Real=1.)
    A_mul_B!(out, p.X, f == 0 ? p.beta0 : broadcast!(muladd, p.scratchbeta, f, p.delbeta, p.beta0))
end
linpred(p::LinPred, f::Real=1.) = linpred!(Array(eltype(p.X), size(p.X, 1)), p, f)

## Install beta0 + f*delbeta as beta0 and zero out delbeta
function installbeta!(p::LinPred, f::Real=1.)
    beta0 = p.beta0
    delbeta = p.delbeta
    @inbounds for i = eachindex(beta0,delbeta)
        beta0[i] += delbeta[i]*f
        delbeta[i] = 0
    end
    p.beta0
end

@compat typealias BlasReal Union{Float32,Float64}

type DensePredQR{T<:BlasReal} <: DensePred
    X::Matrix{T}                  # model matrix
    beta0::Vector{T}              # base coefficient vector
    delbeta::Vector{T}            # coefficient increment
    scratchbeta::Vector{T}
    qr::QRCompactWY{T}
    function DensePredQR(X::Matrix{T}, beta0::Vector{T})
        n, p = size(X)
        length(beta0) == p || throw(DimensionMismatch("length(β0) ≠ size(X,2)"))
        new(X, beta0, zeros(T,p), zeros(T,p), qrfact(X))
    end
end
convert{T}(::Type{DensePredQR{T}}, X::Matrix{T}) = DensePredQR{T}(X, zeros(T, size(X, 2)))

delbeta!{T<:BlasReal}(p::DensePredQR{T}, r::Vector{T}) = (p.delbeta = p.qr\r; p)

type DensePredChol{T<:BlasReal,C} <: DensePred
    X::Matrix{T}                   # model matrix
    beta0::Vector{T}               # base vector for coefficients
    delbeta::Vector{T}             # coefficient increment
    scratchbeta::Vector{T}
    chol::C
    scratch::Matrix{T}
end
DensePredChol{T<:BlasReal}(X::Matrix{T}) =
    DensePredChol(X, zeros(T, size(X, 2)), zeros(T, size(X, 2)), zeros(T, size(X, 2)), cholfact!(X'X), similar(X))
cholpred(X::Matrix) = DensePredChol(X)

cholfactors(c::Cholesky) = c.factors
Base.LinAlg.cholfact!{T<:FP}(p::DensePredChol{T}) = p.chol
Base.LinAlg.cholfact{T<:FP}(p::DensePredQR{T}) = Cholesky(copy(p.qr[:R]), 'U')
Base.LinAlg.cholfact{T<:FP}(p::DensePredChol{T}) = (c = p.chol; Cholesky(copy(cholfactors(c)), c.uplo))
Base.LinAlg.cholfact!{T<:FP}(p::DensePredQR{T}) = Cholesky(p.qr[:R], 'U')

function delbeta!{T<:BlasReal}(p::DensePredChol{T}, r::Vector{T})
    A_ldiv_B!(p.chol, At_mul_B!(p.delbeta, p.X, r))
    p
end

function delbeta!{T<:BlasReal}(p::DensePredChol{T}, r::Vector{T}, wt::Vector{T})
    scr = scale!(p.scratch, wt, p.X)
    cholfact!(At_mul_B!(cholfactors(p.chol), scr, p.X), :U)
    A_ldiv_B!(p.chol, At_mul_B!(p.delbeta, scr, r))
    p
end

type SparsePredChol{T,M<:SparseMatrixCSC,C} <: GLM.LinPred
    X::M                           # model matrix
    Xt::M                          # X'
    beta0::Vector{T}               # base vector for coefficients
    delbeta::Vector{T}             # coefficient increment
    scratchbeta::Vector{T}
    chol::C
    scratch::M
end
function SparsePredChol{T}(X::SparseMatrixCSC{T})
    chol = cholfact(speye(size(X, 2)))
    SparsePredChol{eltype(X),typeof(X),typeof(chol)}(X, X', zeros(T, size(X, 2)), zeros(T, size(X, 2)), zeros(T, size(X, 2)), chol, similar(X))
end
cholpred(X::SparseMatrixCSC) = SparsePredChol(X)

function delbeta!{T}(p::SparsePredChol{T}, r::Vector{T}, wt::Vector{T})
    scr = scale!(p.scratch, wt, p.X)
    XtWX = p.Xt*scr
    c = p.chol = cholfact(Symmetric{eltype(XtWX),typeof(XtWX)}(XtWX, 'L'))
    p.delbeta = c\Ac_mul_B!(p.delbeta, scr, r)
end

Base.cholfact{T}(p::SparsePredChol{T}) = copy(p.chol)
Base.cholfact!{T}(p::SparsePredChol{T}) = p.chol

coef(x::LinPred) = x.beta0
coef(x::LinPredModel) = coef(x.pp)

df_residual(x::LinPredModel) = df_residual(x.pp)
df_residual(x::@compat(Union{DensePred,SparsePredChol})) = size(x.X, 1) - length(x.beta0)

invchol(x::DensePred) = inv(cholfact!(x))
invchol(x::SparsePredChol) = cholfact!(x)\eye(size(x.X, 2))
vcov(x::LinPredModel) = scale!(invchol(x.pp), scale(x,true))

function cor(x::LinPredModel)
    Σ = vcov(x)
    invstd = similar(Σ, size(Σ, 1))
    for i = eachindex(invstd)
        invstd[i] = 1/sqrt(Σ[i, i])
    end
    scale!(invstd, scale!(Σ, invstd))
end

stderr(x::LinPredModel) = sqrt(diag(vcov(x)))

function show(io::IO, obj::LinPredModel)
    println(io, "$(typeof(obj)):\n\nCoefficients:\n", coeftable(obj))
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
