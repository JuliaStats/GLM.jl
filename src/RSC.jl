## Regular sparse column-oriented matrices

## The representation is as two dense matrices the non-zero values and the row indices
## This requires that the number of nonzeros in each column be constant
## The row indices should be sorted within columns

require("sparse.jl")

type SparseMatrixRSC{Tv,Ti<:Union(Int32,Int64)} <: AbstractMatrix{Tv}
    m::Int                              # number of rows
    rowval::Matrix{Ti}                  # row indices of nonzeros
    nzval::Matrix{Tv}                   # nonzero values
    function SparseMatrixRSC(rowval::Matrix{Ti},nzval::Matrix{Tv})
        m,n = size(rowval)
        if size(nzval) != size(rowval)
            error("sizes of nzval, $(size(nzval)), and rowval, $(size(rowval)), must match")
        end
        if !all(rowval .> 0) error("row values must be positive") end
        new(max(rowval), rowval, nzval)
    end
end

function SparseMatrixRSC(rowval::Matrix,nzval::Matrix)
    SparseMatrixRSC{eltype(nzval),eltype(rowval)}(rowval,nzval)
end

issparse(A::SparseMatrixRSC) = true
nnz(A::SparseMatrixRSC) = numel(A.nzval)

size{Tv,Ti}(A::SparseMatrixRSC{Tv,Ti}) = (A.m, size(A.nzval,2))
size{Tv,Ti}(A::SparseMatrixRSC{Tv,Ti},d) = d == 1 ? A.m : size(A.nzval,d)
copy(A::SparseMatrixRSC) = SparseMatrixRSC(A.m, copy(A.rowval), copy(A.nzval))
function convert{Tv,Ti}(::Type{SparseMatrixCSC}, x::SparseMatrixRSC{Tv,Ti})
    k,n = size(x.rowval)
    m   = max(x.rowval)
    nnz = k * n
    SparseMatrixCSC{Tv,Ti}(m, n, 1 + [x for x in 0:k:(k*n)],
                           reshape(copy(x.rowval), (nnz,)),
                           reshape(copy(x.nzval), (nnz,)))
end

convert{Tv}(::Type{Matrix}, x::SparseMatrixRSC{Tv}) = convert(Matrix{Tv}, convert(SparseMatrixCSC, x))

show(io, A::SparseMatrixRSC) = show(io, convert(SparseMatrixCSC, A))

function tcrossprod!(A::SparseMatrixRSC, C::SparseMatrixCSC)
    m,n = size(A)
    k,j = size(C)
    if j != k error("C must be square but size(C) is $(size(C))") end
    if j != m error("Dimension mismatch") end
    C.nzval[:] = zero(eltype(C))
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
