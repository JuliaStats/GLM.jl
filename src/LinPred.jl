abstract LinPred                        # linear predictor for statistical models
linPred(p::LinPred) = p.X * p.beta      # general calculation of the linear predictor

type DensePred{T<:Number} <: LinPred    # predictor with dense X
    X::Matrix{T}                        # model matrix
    beta::Vector{T}                     # coefficient vector
    qr::QRDense{T}
    function DensePred(X::Matrix{T}, beta::Vector{T})
        n, p = size(X)
        if length(beta) != p error("dimension mismatch") end
        new(X, beta, qr(X))
    end
end

## outer constructor
DensePred(X::Matrix{Float64}) = DensePred{Float64}(X, zeros(Float64,(size(X,2),)))
DensePred{T<:Real}(X::Matrix{T}) = DensePred(float64(X))

function updateBeta(p::DensePred, y::Vector{Float64}, sqrtwt::Vector{Float64})
    p.qr = qr(diagmm(sqrtwt, p.X))
    p.beta = p.qr \ (sqrtwt .* y)
end

At_mult_B{T <: Real}(A::DArray{T, 2, 1}, B::DArray{T, 2, 1}) = Ac_mult_B(A, B)

function Ac_mult_B{T <: Real}(A::DArray{T, 2, 1}, B::DArray{T, 2, 1})
    if (all(procs(A) != procs(B)) || all(dist(A) != dist(B)))
                                        # FIXME: B should be redistributed to match A
        error("Arrays A and B must be distributed similarly")
    end
    if is(A, B)
        return mapreduce(+, fetch, {@spawnat p _jl_syrk('T', localize(A)) for p in procs(A)})
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
        new(X, beta, CholeskyDense(X'*X))
    end
end

function (\)(A::DArray{Float64,2,1}, B::DArray{Float64,1,1})
    R   = CholeskyDense(A' * A)              # done by _jl_dsyrk, see above
    AtB = A' * B
    n = size(A, 2)
    info = Array(Int32, 1)
    one = 1
    ccall(dlsym(_jl_liblapack, :dpotrs_), Void,
          (Ptr{Uint8}, Ptr{Int32}, Ptr{Int32}, Ptr{Float64}, Ptr{Int32},
          Ptr{Float64}, Ptr{Int32}, Ptr{Int32}),
          "U", &n, &one, R, &n, AtB, &n, info)
    if info == 0; return AtB; end
    if info > 0; error("matrix not positive definite"); end
    error("error in CHOL")
end

