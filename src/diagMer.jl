require("linalg_suitesparse.jl")

## move this to linalg_suitesparse.jl, perhaps under another name
function aat{Tv<:CHMVTypes,Ti<:CHMITypes}(A::SparseMatrixCSC{Tv,Ti})
    cs = CholmodSparse(A, 0)
    aa = CholmodPtr{Tv,Ti}(Array(Ptr{Void},1))
    aa.val[1] = ccall(dlsym(_jl_libcholmod, :cholmod_aat), Ptr{Void},
                  (Ptr{Void},Ptr{Int},Int32,Int32,Ptr{Void}),
                  cs.pt.val[1], C_NULL, 0, 1, cs.cm.pt[1])
    status = ccall(dlsym(_jl_libcholmod, :cholmod_sort), Int32,
                   (Ptr{Void}, Ptr{Void}), aa.val[1], cs.cm.pt[1])
    if status != 1 error("Error calling cholmod_l_sort") end
    m  = size(A, 1)
    CholmodSparseOut{Tv,Ti}(aa, m, m, cs.cm)
end

## a mixed-effects representation for models with simple, scalar random effects only
type diagMer{Tv<:CHMVTypes, Ti<:CHMITypes}
    Z::SparseMatrixCSC{Tv,Ti}
    X::Matrix{Tv}
    theta::Vector{Tv}
    Lind::Vector{Ti}
    L::CholmodFactor{Tv,Ti}
    ZXty::Vector{Tv}
    levs
end

function diagMer{Tv<:CHMVTypes}(X::Matrix{Tv}, y::Vector{Tv}, factors::Vector...)
    nfac = length(factors)
    mm   = map(f->indicators(f, true), factors)
    Z    = reduce(hcat, map(x->x[1], mm))
    n, q = size(Z)
    levs = map(x->x[2], mm)
    Lind = reduce(vcat, map(i->fill(i, length(levs[i]),), int(1:length(levs))))
    p    = size(X, 2)
    (I, J, V) = findn_nzs(X)
    Ti   = eltype(Z.colptr)
    ZXt  = hcat(Z, _jl_sparse_sorted!(convert(Vector{Ti},I), convert(Vector{Ti}, J), V, n, p, +))'
    cs   = aat(ZXt)
    ind  = convert(Vector{Ti}, int(0:(q-1)))
    a    = Array(Ptr{Void}, 1)
    a[1] = ccall(dlsym(_jl_libcholmod, :cholmod_submatrix), Ptr{Void},
                 (Ptr{Void}, Ptr{Int32}, Int, Ptr{Int32}, Int, Int32, Int32, Ptr{Void}),
                 cs.pt.val[1], ind, q, ind, q, 1, 1, cs.cm.pt[1])
    b    = Array(Ptr{Void}, 1)
    b[1] = ccall(dlsym(_jl_libcholmod, :cholmod_copy), Ptr{Void},
                 (Ptr{Void}, Int32, Int32, Ptr{Void}), a[1], 1, 1, cs.cm.pt[1])
    st   = ccall(dlsym(_jl_libcholmod, :cholmod_free_sparse), Int32,
                 (Ptr{Void}, Ptr{Void}), a, cs.cm.pt[1])
    if st != 1 error("CHOLMOD error in free_sparse") end
    ord  = Array(Ti, q + p)
    ind  = q + (1:p)
    ord[ind] = ind - 1                  # 0-based indices
    st   = ccall(dlsym(_jl_libcholmod, :cholmod_amd), Ti,
                 (Ptr{Void}, Ptr{Int32}, Uint, Ptr{Ti}, Ptr{Void}),
                 b[1], int32(0:(q-1)), q, ord, cs.cm.pt[1])
    if st != 1 error("CHOLMOD error in amd") end
    st   = ccall(dlsym(_jl_libcholmod, :cholmod_free_sparse), Int32,
                 (Ptr{Void}, Ptr{Void}), b, cs.cm.pt[1])
    if st != 1 error("CHOLMOD error in free_sparse") end
    pt   = CholmodPtr{Tv,Ti}(Array(Ptr{Void}, 1))
    pt.val[1] = ccall(dlsym(_jl_libcholmod, :cholmod_analyze_p), Ptr{Void},
                      (Ptr{Void}, Ptr{Ti}, Ptr{Void}, Uint, Ptr{Void}),
                      cs.pt.val[1], ord, C_NULL, 0, cs.cm.pt[1])
    st   = ccall(dlsym(_jl_libcholmod, :cholmod_factorize), Int32,
                 (Ptr{Void}, Ptr{Void}, Ptr{Void}), cs.pt.val[1], pt.val[1], cs.cm.pt[1])
    if st != 1 error("CHOLMOD failure in factorize") end
    return CholmodFactor{Tv,Ti}(pt, cs)
    diagMer{Tv,Ti}(Z, X, ones(Tv, length(levs)), Lind, CholmodFactor{Tv,Ti}(pt, cs), ZXt * y)
end

function update{Tv<:CHMVTypes,Ti<:CHMITypes}(dd::diagMer{Tv,Ti}, theta::Vector{Tv})
    if length(theta) != length(dd.theta) error("Dimension mismatch") end
    di = theta[Lind]
    sp = SparseMatrixCSC(dd.L.cs)
    rv = sp.rowval
    nz = sp.nzval
    cp = sp.colptr
    for j = 1:length(di), k = cp[j]:(cp[j+1] - 1)
        rr = rv[k]
        nz[k] *= di[rr] * di[j]
        if rr == j nz[k] += 1. end
    end
    cs = CholmodSparse(sp, 1)
    st = ccall(dlsym(_jl_libcholmod, :cholmod_factorize), Int32,
               (Ptr{Void}, Ptr{Void}, Ptr{Void}), cs.pt.val[1], dd.L.pt.val[1], cs.cm.pt[1])
    if st != 1 error("CHOLMOD failure in factorize") end
    dd.L \ dd.ZXty
end

y = convert(Vector{Float64},[1545,1440,1440,1520,1580,1540,1555,1490,1560,1495,1595,1550,1605,1510,1560,1445,1440,1595,1465,1545,1595,1630,1515,1635,1625,1520,1455,1450,1480,1445])

Lind = ones(Int32, 6)

X = ones((length(y),1))

function pattern(io::IO, A::SparseMatrixCSC)
    At = A'
    rowptr = At.colptr
    colval = At.rowval
    m, n = size(A)
    println(io)
    for i in 1:m
        v = falses(n)
        for k in rowptr[i]:(rowptr[i + 1] - 1) v[colval[k]] = true end
        for j in 1:n print(io, v[j] ? "| " : ". ") end
        println(io)
    end
end

pattern(A::SparseMatrixCSC) = pattern(stdout_stream, A)
    
pattern(io::IO, A::CholmodSparseOut) = pattern(io, SparseMatrixCSC(A))
pattern(A::CholmodSparseOut) = pattern(stdout_stream, SparseMatrixCSC(A))
