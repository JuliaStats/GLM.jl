require("linalg_suitesparse.jl")

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

const _chm_analyze_p = dlsym(_jl_libcholmod, :cholmod_analyze_p)

function diagMer{Tv<:CHMVTypes,Ti<:CHMITypes}(X::Matrix{Tv}, y::Vector{Tv}, factors::Vector{Ti}...)
    nfac = length(factors)
    mm   = map(f->indicators(f, true), factors)
    Z    = reduce(hcat, map(x->x[1], mm))
    n, q = size(Z)
    levs = map(x->x[2], mm)
    Lind = convert(Vector{Ti}, reduce(vcat, map(i->fill(i, length(levs[i]),), int(1:length(levs)))))
    p    = size(X, 2)
    (I, J, V) = findn_nzs(X)
    ZXt  = hcat(Z, _jl_sparse_sorted!(convert(Vector{Ti},I), convert(Vector{Ti}, J), V, n, p, +))'
    cs   = chm_aat(ZXt)
                                        # extract the q by q upper left submatrix
    ind  = convert(Vector{Ti}, int(0:(q-1)))
    a    = Array(Ptr{Void}, 1)
    a[1] = ccall(_chm_submatrix, Ptr{Void},
                 (Ptr{Void}, Ptr{Int32}, Int, Ptr{Int32}, Int, Int32, Int32, Ptr{Void}),
                 cs.pt.val[1], ind, q, ind, q, 1, 1, cs.cm.pt[1])
                                        # convert to (upper) symmetric storage
    b    = Array(Ptr{Void}, 1)
    b[1] = ccall(_chm_copy, Ptr{Void}, (Ptr{Void}, Int32, Int32, Ptr{Void}), a[1], 1, 1, cs.cm.pt[1])
    st   = ccall(_chm_free_sp, Int32, (Ptr{Void}, Ptr{Void}), a, cs.cm.pt[1])
    if st != 1 error("CHOLMOD error in free_sparse") end
                                        # create the permutation vector with identity beyond q
    ord  = Array(Ti, q + p)
    ind  = q + (1:p)
    ord[ind] = ind - 1                  # 0-based indices
                                        # use amd on q by q matrix
    st   = ccall(_chm_amd, Int32, (Ptr{Void}, Ptr{Int32}, Uint, Ptr{Ti}, Ptr{Void}),
                 b[1], C_NULL, 0, ord, cs.cm.pt[1])
    if st != 1 error("CHOLMOD error in amd") end
                                        # free the q by q matrix
    st   = ccall(_chm_free_sp, Int32, (Ptr{Void}, Ptr{Void}), b, cs.cm.pt[1])
    if st != 1 error("CHOLMOD error in free_sparse") end
                                        # convert cs to (upper) symmetric storage
    b[1] = ccall(_chm_copy, Ptr{Void}, (Ptr{Void}, Int32, Int32, Ptr{Void}), cs.pt.val[1], 1, 1, cs.cm.pt[1])
    st   = ccall(_chm_free_sp, Int32, (Ptr{Void}, Ptr{Void}), cs.pt.val, cs.cm.pt[1])
    if st != 1 error("CHOLMOD error in free_sparse") end
    cs.pt.val[1] = b[1]
                                        # Create CholmodFactor - FIXME: set nmethods = 1
    ptf  = CholmodPtr{Tv,Ti}(Array(Ptr{Void},1))
    ptf.val[1] = ccall(_chm_analyze_p, Ptr{Void}, (Ptr{Void}, Ptr{Ti}, Ptr{Void}, Uint, Ptr{Void}),
                       cs.pt.val[1], ord, C_NULL, 0, cs.cm.pt[1])
    bb  = CholmodSparse{Tv,Ti}(SparseMatrixCSC(cs), 1, cs.cm)
    diagMer{Tv,Ti}(Z, X, ones(Tv, length(levs)), Lind, CholmodFactor{Tv,Ti}(ptf, bb), ZXt * y, levs)
end

function update{Tv<:CHMVTypes,Ti<:CHMITypes}(dd::diagMer{Tv,Ti}, theta::Vector{Tv})
    if length(theta) != length(dd.theta) error("Dimension mismatch") end
    di = theta[dd.Lind]
    sp = SparseMatrixCSC(dd.L.cs)
    rv = sp.rowval
    nz = sp.nzval
    cp = sp.colptr
    for j = 1:length(di), k = cp[j]:(cp[j+1] - 1)
        rr = rv[k]
        nz[k] *= di[rr] * di[j]
        if rr == j nz[k] += 1. end
    end
    cs = CholmodSparse{Tv,Ti}(sp, 1, dd.L.cs.cm)
    st = ccall(_chm_factorize, Int32, (Ptr{Void}, Ptr{Void}, Ptr{Void}),
               cs.pt.val[1], dd.L.pt.val[1], cs.cm.pt[1])
    if st != 1 error("CHOLMOD failure in factorize") end
    bb = solve(dd.L, CholmodDense(dd.ZXty, cs.cm))
end

#y = convert(Vector{Float64},[1545,1440,1440,1520,1580,1540,1555,1490,1560,1495,1595,1550,1605,1510,1560,1445,1440,1595,1465,1545,1595,1630,1515,1635,1625,1520,1455,1450,1480,1445])

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
