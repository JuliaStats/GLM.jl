abstract LinPred                        # linear predictor for statistical models
linPred(p::LinPred) = p.X * p.beta

type DensePred <: LinPred                   # predictor with dense X
    X::Matrix{Float64}                  # model matrix
    beta::Vector{Float64}               # coefficient vector
    DensePred(Xi, bi) = size(Xi, 2)==size(bi, 1)? new(float64(Xi), float64(bi)) : error("dimension mismatch")
end

## outer constructor
DensePred(X::Matrix{Float64}) = DensePred(X, zeros(Float64,(size(X, 2),)))

function updateBeta(p::DensePred, y::Vector{Float64}, sqrtwt::Vector{Float64})
    p.beta, R, wrss = wtdLS(p.X, y, sqrtwt)
    p
end

function wtdLS(X::StridedMatrix{Float64}, y::VecOrMat{Float64}, sqrtwt::Vector{Float64})
    m, n  = size(X)
    if (m != size(y, 1)) || (m != size(sqrtwt, 1)) error("Dimension mismatch") end
    if m < n error("Underdetermined system (m < n)") end
    
    QR    = diagmm(sqrtwt, X)
    qty   = diagmm(sqrtwt, isa(y, Vector) ? reshape(y, (size(y, 1), 1)) : y)
    nrhs  = size(qty, 2)
    work  = Array(Float64, 1)
    lwork = int32(-1)
    if 0 ==_jl_lapack_gels("N", m, n, nrhs, QR, stride(QR, 2), qty, m, work, lwork)
        lwork = int32(work[1])
        work  = Array(Float64, int(lwork))
    else
        error("error in LAPACK gels")
    end

    if 0 != _jl_lapack_gels("N", m, n, nrhs, QR, m, qty, m, work, lwork)
        error("error in LAPACK gels")
    end
    beta  = isa(y, Vector) ? reshape(qty[1:n,:], (n,)) : qty[1:n,:]
    (beta, triu(QR[1:n,:]), [sum(qty[(n + 1):m, i].^2) for i=1:size(qty,2)])
end

type DistPred <: LinPred                # predictor with distributed (on rows) X
    X::DArray{Float64, 2, 1}            # model matrix
    beta::Vector{Float64}               # coefficient vector
    DistPred(Xi, bi) = size(Xi, 2)==size(bi, 1)? new(float64(Xi), float64(bi)) : error("dimension mismatch")
end

function (\)(A::DArray{Float64,2,1}, B::DArray{Float64,1,1})
    if (all(procs(A) != procs(B)) || all(dist(A) != dist(B)))
        error("Arrays A and B must be distributed similarly")
    end
    R   = chol(mapreduce(+, fetch, {@spawnat p _jl_syrk('T', localize(A)) for p in procs(A)}))
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

