@compat typealias BlasReal Union{Float32,Float64}

abstract QR{T} <: LinFact{T}

"Dense QR factorization (using DenseQR), unweighted"
type DenseQRUnweighted{T} <: QR{T}
    X::Matrix{T}                  # model matrix
    scratch::Matrix{T}            # scratch for QR
    qr::QRCompactWY{T,Matrix{T}}  # QR object
    DenseQRUnweighted(X::Matrix{T}) = new(X, similar(X))
end
DenseQRUnweighted{T<:BlasReal}(X::Matrix{T}) = DenseQRUnweighted{T}(X)

"Computes the QR factorization of X"
factorize!{T}(p::DenseQRUnweighted{T}) =
    (p.qr = qrfact!(copy!(p.scratch, p.X)); p)

Base.(:\){T}(p::DenseQRUnweighted{T}, r::AbstractVector{T}) =
    p.qr\r


"Dense QR factorization (using LAPACK), weighted"
type DenseQRWeighted{T} <: QR{T}
    X::Matrix{T}                  # model matrix
    scratch::Matrix{T}            # scratch for QR
    sqrtwt::Vector{T}             # sqrt of weights
    rscratch::Vector{T}           # scratch for computation of r.*sqrt(wt)
    qr::QRCompactWY{T,Matrix{T}}  # QR object
    DenseQRWeighted(X::Matrix{T}) = new(X, similar(X), similar(X, size(X, 1)), similar(X, size(X, 1)))
end
DenseQRWeighted{T<:BlasReal}(X::Matrix{T}) = DenseQRWeighted{T}(X)

"Computes the QR factorization of WX where W = diagm(sqrt(wt))"
function factorize!{T}(p::DenseQRWeighted{T}, wt::AbstractVector{T})
    broadcast!(sqrt, p.sqrtwt, wt)
    scale!(p.scratch, p.sqrtwt, p.X)
    p.qr = qrfact!(p.scratch)
    p
end

Base.(:\){T}(p::DenseQRWeighted{T}, r::AbstractVector{T}) =
    p.qr\broadcast!(*, p.rscratch, r, p.sqrtwt)

Base.LinAlg.cholfact!{T<:FP}(p::QR{T}) =
    Cholesky(p.qr[:R], 'U')



abstract Chol{T} <: LinFact{T}

"Dense Cholesky factorization, unweighted"
immutable DenseCholUnweighted{T<:BlasReal,C} <: Chol{T}
    X::Matrix{T}                   # model matrix
    chol::C                        # Cholesky object
end
DenseCholUnweighted{T<:BlasReal}(X::Matrix{T}) = 
    DenseCholUnweighted(X, Cholesky(Array(T, size(X, 2), size(X, 2)), 'U'))

"Computes the Cholesky factorization of X'X"
factorize!{T}(p::DenseCholUnweighted{T}) =
    (cholfact!(At_mul_B!(p.chol.factors, p.X, p.X), :U); p)

solve!{T}(beta::StridedVector{T}, p::DenseCholUnweighted{T}, r::AbstractVector{T}) =
    A_ldiv_B!(p.chol, At_mul_B!(beta, p.X, r))


"Dense Cholesky factorization, weighted"
immutable DenseCholWeighted{T<:BlasReal,C} <: Chol{T}
    X::Matrix{T}                   # model matrix
    WX::Matrix{T}                  # W*X
    chol::C                        # Cholesky object
end
DenseCholWeighted{T<:BlasReal}(X::Matrix{T}) =
    DenseCholWeighted(X, similar(X), Cholesky(Array(T, size(X, 2), size(X, 2)), 'U'))

"Computes the Cholesky factorization of X'WX where W = diagm(wt)"
function factorize!{T}(p::DenseCholWeighted{T}, wt::AbstractVector{T})
    scale!(p.WX, wt, p.X)
    cholfact!(At_mul_B!(p.chol.factors, p.WX, p.X), :U)
    p
end

solve!{T}(beta::StridedVector{T}, p::DenseCholWeighted{T}, r::AbstractVector{T}) =
    A_ldiv_B!(p.chol, At_mul_B!(beta, p.WX, r))



"Sparse Cholesky factorization using CHOLMOD"
type SparseChol{T,M<:SparseMatrixCSC} <: Chol{T}
    X::M                           # model matrix
    Xt::M                          # X'
    WX::M                          # WX
    chol::Base.SparseMatrix.CHOLMOD.Factor{T}
    SparseChol(X) = new(X, X', similar(X))
end
SparseChol{T}(X::SparseMatrixCSC{T}) = SparseChol{T,typeof(X)}(X)

"Computes the Cholesky factorization of X'X"
function factorize!{T}(p::SparseChol{T})
    p.WX = p.X
    p.chol = cholfact(Symmetric{T,typeof(p.X)}(p.Xt*p.X, 'U'))
    p
end

"Computes the Cholesky factorization of X'WX where W = diagm(wt)"
function factorize!{T}(p::SparseChol{T}, wt::Vector{T})
    scr = scale!(p.WX, wt, p.X)
    XtX = p.Xt*scr
    p.chol = cholfact(Symmetric{eltype(XtX),typeof(XtX)}(XtX, 'U'))
    p
end

solve!{T}(beta::StridedVector{T}, p::SparseChol{T}, r::AbstractVector{T}) =
    p.chol\Ac_mul_B!(beta, p.WX, r)

Base.LinAlg.cholfact!{T<:FP}(p::Chol{T}) = p.chol



# We assume one of solve or solve! is defined for each LinFact.
# Otherwise you will get a StackOverflowError.
"""
Solves the least squares problem after the factorization has been
computed using factorize! and returns the results in the first argument

Unlike A_ldiv_B!, this may use the vector provided as the first
argument as the output, or it may return a new vector.
"""
solve!{T}(out::StridedVector{T}, p::LinFact{T}, beta::AbstractVector) = p\beta

"""
Solves the least squares problem after the factorization has been
computed using factorize!
"""
Base.(:\){T}(p::LinFact{T}, beta::AbstractVector) = solve!(Array(T, size(p.X, 2)), p, beta)

Base.A_mul_B!(out::AbstractVector, p::LinFact, beta::AbstractVector) =
    A_mul_B!(out, p.X, beta)

df_residual(x::LinPredModel) = df_residual(x.pp)
df_residual(x::LinFact) = size(x.X, 1) - size(x.X, 2)

invchol(x::LinFact) = inv(cholfact!(x))
invchol(x::SparseChol) = cholfact!(x)\eye(size(x.X, 2))
vcov(x::LinPredModel) = scale!(invchol(x.pp), scale(x,true))
#vcov(x::DensePredChol) = inv(x.chol)
#vcov(x::DensePredQR) = copytri!(potri!('U', x.qr[:R]), 'U')

function cor(x::LinPredModel)
    Σ = vcov(x)
    invstd = similar(Σ, size(Σ, 1))
    for i = 1:size(Σ, 1)
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

ModelMatrix(obj::LinPredModel) = obj.pp.X
model_response(obj::LinPredModel) = obj.rr.y

fitted(m::LinPredModel) = m.rr.mu
predict(mm::LinPredModel) = fitted(mm)
formula(obj::LinPredModel) = ModelFrame(obj).formula
nobs(obj::LinPredModel) = length(model_response(obj))
residuals(obj::LinPredModel) = residuals(obj.rr)
