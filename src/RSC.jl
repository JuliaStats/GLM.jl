## Regular sparse column-oriented matrices

## The representation is as two dense matrices the non-zero values and
## the row indices.  This requires that the number of nonzeros in each
## column be constant.  The row indices should be sorted within columns

require("sparse.jl")
import Base.*                           # to define a new method

type SparseMatrixRSC{Tv,Ti<:Union(Int32,Int64)} <: AbstractSparseMatrix{Tv,Ti}
    m::Int                              # number of rows
    rowval::Matrix{Ti}                  # row indices of nonzeros
    nzval::Matrix{Tv}                   # nonzero values
    function SparseMatrixRSC(rowval::Matrix{Ti},nzval::Matrix{Tv})
        m,n = size(rowval)
        if size(nzval) != size(rowval)
            error("sizes of nzval, $(size(nzval)), and rowval, $(size(rowval)), must match")
        end
        if !all(rowval .> 0) error("row values must be positive") end
        new(int(max(rowval)), rowval, nzval)
    end
end

function SparseMatrixRSC(rowval::Matrix,nzval::Matrix)
    SparseMatrixRSC{eltype(nzval),eltype(rowval)}(rowval,nzval)
end

nnz(A::SparseMatrixRSC) = length(A.nzval)

size(A::SparseMatrixRSC) = (A.m, size(A.nzval,2))
size(A::SparseMatrixRSC,d) = d == 1 ? A.m : size(A.nzval,d)
copy(A::SparseMatrixRSC) = SparseMatrixRSC(copy(A.rowval), copy(A.nzval))

function convert{Tv,Ti}(::Type{SparseMatrixCSC}, x::SparseMatrixRSC{Tv,Ti})
    k,n = size(x.rowval)
    m   = int(max(x.rowval))
    nnz = k * n
    SparseMatrixCSC{Tv,Ti}(m, n,
                           convert(Ti,1) + [convert(Ti,x) for x in 0:k:(k*n)],
                           reshape(copy(x.rowval), (nnz,)),
                           reshape(copy(x.nzval), (nnz,)))
end

convert{Tv}(::Type{Matrix}, x::SparseMatrixRSC{Tv}) = convert(Matrix{Tv}, convert(SparseMatrixCSC, x))

function show(io, A::SparseMatrixRSC)
    println(io, "$(A.m) by $(size(A.nzval, 2)) regular sparse column matrix")
    println(io, "Row indices: ", A.rowval)
    print(io, "Non-zero values: ", A.nzval)
end

function tcrossprod!(A::SparseMatrixRSC, C::SparseMatrixCSC)
    m,n = size(A)
    k,j = size(C)
    if j != k error("C must be square but size(C) is $(size(C))") end
    if j != m error("Dimension mismatch") end
    Cx = C.nzval
    Cr = C.rowval
    Cc = C.colptr
    Ax = A.nzval
    Ar = A.rowval
    kk = size(Ax, 1)
    for j in 1:n
        for k in 1:kk
            jj = Ar[k,j]
            x  = Ax[k,j]
            ii = Cc[jj]
            for l = k:kk
                ll = Ar[l,j]
                while ii < Cc[jj+1] && Cr[ii] < ll ii += 1 end
                if Cr[ii] != ll error("column $i in C does not have a non-zero at row $ll") end
                C.nzval[ii] += x * Ax[l,j]
            end
        end
    end
    C
end

function tcrossprodPat{Tv,Ti}(A::SparseMatrixRSC{Tv,Ti}, dd)
    m,n = size(A)
    Ar  = A.rowval
    kk  = size(Ar,1)
    vv  = [Set{Ti}() for i in 1:m]
    for j in 1:n
        for k in 1:kk
            jj = Ar[k,j]
            ss = vv[jj]
            for l in k:kk
                ll = Ar[l,j]
                if !has(ss, ll) add(ss, ll) end
            end
        end
    end
    colptr = one(Ti) + vcat(zero(Ti), cumsum([convert(Ti,length(s)) for s in vv]))
    rowval = mapreduce(vcat, s->sort!(elements(s)), vv)
    aat = SparseMatrixCSC{Tv,Ti}(m, m, colptr, rowval, zeros(Tv, (length(rowval,))))
    for d in dd, i in colptr[d]:(colptr[d+1] - 1)
        if rowval[i] == d
            aat.nzval[i] = one(Tv)
            break
        end
    end
    aat
end

tcrossprodPat{Tv,Ti}(A::SparseMatrixRSC{Tv,Ti}) = tcrossprodPat(A, Array(Ti, (0,)))

## DyeStuff example from the lme4 package for R
#ZXt = SparseMatrixRSC(int32(hcat(Glm.gl(6,5), 7.*ones(Int, 30)))', ones(Float64,(2,30)))
#ZXtZX = tcrossprodPat(ZXt,1:6)
#tcrossprod!(ZXt, copy(ZXtZX))
#Yield = float([1545, 1440, 1440, 1520, 1580, 1540, 1555, 1490, 1560, 1495, 1595, 1550, 1605, 1510, 1560, 1445, 1440, 1595, 1465, 1545, 1595, 1630, 1515, 1635, 1625, 1520, 1455, 1450, 1480, 1445])

function scale!{T}(sc::Vector{T}, A::SparseMatrixRSC{T})
    if length(sc) != size(A.nzval, 1) error("Dimension mismatch") end
    diagmm!(A.nzval, sc, A.nzval)
    A
end

function *{T}(A::SparseMatrixRSC{T}, v::Vector{T})
    m,n = size(A)
    if length(v) != n error("Dimension mismatch") end
    rv  = A.rowval
    nv  = A.nzval
    k   = size(nv, 1)
    res = zeros(T, m)
    for j in 1:n, i in 1:k
        res[rv[i,j]] += v[j] * nv[i,j]
    end
    res
end


