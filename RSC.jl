## Regular sparse column-oriented matrices

## The representation is as two dense matrices the non-zero values and
## the row indices.  This requires that the number of nonzeros in each
## column be constant.  The row indices should be sorted within columns

## To allow for both random effects and fixed-effects terms in a mixed-effects
## model the non-zero values can have more rows than the row indices, the
## convention being that the p extra rows are dense rows appended to the matrix.

type SparseMatrixRSC{Tv<:VTypes,Ti<:ITypes} <: AbstractSparseMatrix{Tv,Ti}
    q::Int                              # number of rows in the Zt part
    p::Int                              # number of rows in the Xt part
    rowval::Matrix{Ti}                  # row indices of nonzeros
    rowrange::Array{Range1{Int},1}       # ranges of rows of rowval
    nzval::Matrix{Tv}                   # nonzero values
    function SparseMatrixRSC(rowval::Matrix{Ti},nzval::Matrix{Tv})
        if size(nzval,2) != size(rowval,2)
            error("number of columns in nzval, $(size(nzval,2)), should be $(size(rowval,2))")
        end
        if !all(rowval .> 0) error("row values must be positive") end
        k = size(rowval, 1)
        rowrange = Array(Range1{Int}, k)
        for i in 1:k
            mn = rowval[i,1]
            mx = rowval[i,1]
            for j in 2:size(rowval,2)
                vv = rowval[i,j]
                if vv < mn mn = vv end
                if vv > mx mx = vv end
            end
            rowrange[i] = Range1(int(mn), 1 + mx - mn)
        end
        new(int(max(rowval)), size(nzval,1) - k, rowval, rowrange, nzval)
    end
end

function SparseMatrixRSC(rowval::Matrix,nzval::Matrix)
    SparseMatrixRSC{eltype(nzval),eltype(rowval)}(rowval,nzval)
end

function SparseMatrixRSC{Tv<:VTypes,Ti<:Integer}(rowval::Vector{Ti},
                                                 nzval::Matrix{Tv})
    SparseMatrixRSC(transpose(int32(rowval)),nzval)
end

function *{T}(A::SparseMatrixRSC{T}, B::Matrix{T})
    m,n = size(A)
    if size(B,1) != n error("Dimension mismatch") end
    p = size(B,2)
    res = zeros(T, (m, p))
    rv  = A.rowval
    nv  = A.nzval
    k   = size(rv,1)
    for l in 1:p, j in 1:n
        for i in 1:k res[rv[i,j],l] += B[j,l] * nv[i,j] end
        for i in 1:A.p res[A.q + i,l] += B[j,l] * nv[k + i, j] end
    end
    res
end

*{T}(A::SparseMatrixRSC{T}, v::Vector{T}) = vec(A*reshape(v,(length(v),1)))

function Ac_mul_B{T}(A::SparseMatrixRSC{T}, b::Vector{T})
    m,n = size(A)
    if length(b) != m error("Dimension mismatch") end
    res = zeros(T, n)
    rv  = A.rowval
    nv  = A.nzval
    k   = size(rv,1)
    for j in 1:n
        for i in 1:k res[j] += b[rv[i,j]] * nv[i,j] end
        for  i in 1:A.p res[j] += b[A.q + i] * nv[k + i, j] end
    end
    res
end

copy(A::SparseMatrixRSC) = SparseMatrixRSC(copy(A.rowval), copy(A.nzval))

function expandi{Tv,Ti}(A::SparseMatrixRSC{Tv,Ti})
    A.p == 0 ? vec(copy(A.rowval)) :
    vec(vcat(A.rowval, mapreduce(i->fill(convert(Ti,i+A.q), (1,size(A,2))),
                                 vcat, 1:A.p)))
end

function expandj{Tv,Ti}(A::SparseMatrixRSC{Tv,Ti})
    vec(diagmm(ones(Ti,size(A.nzval)), convert(Vector{Ti},[1:size(A,2)])))
end

function expandp{Tv,Ti}(A::SparseMatrixRSC{Tv,Ti})
    one(Ti) + convert(Vector{Ti}, [0:size(A.nzval,1):nnz(A)])
end

function sparse(x::SparseMatrixRSC)
    SparseMatrixCSC(size(x,1), size(x,2), expandp(x), expandi(x), vec(x.nzval))
end

function CholmodSparse(x::SparseMatrixRSC)
    Base.LinAlg.CHOLMOD.CholmodSparse!(sparse(x), 0)
end

function A_mul_Bc{Tv,Ti<:ITypes}(A::SparseMatrixRSC{Tv,Ti},
                                 B::SparseMatrixRSC{Tv,Ti})
    if is(A,B)
        chsp = CholmodSparse(A)
        return A_mul_Bc(chsp,chsp)
    end
    A_mul_Bc(CholmodSparse(A),CholmodSparse(B))
end
    
dense(x::SparseMatrixRSC) = dense(sparse(x))

full(x::SparseMatrixRSC) = dense(sparse(x))

nnz(A::SparseMatrixRSC) = length(A.nzval)

size(A::SparseMatrixRSC) = (A.p + A.q, size(A.nzval,2))
size(A::SparseMatrixRSC,d) = d == 1 ? A.p + A.q : size(A.nzval,d)

function show(io::IO, A::SparseMatrixRSC)
    println(io, "$(size(A,1)) by $(size(A, 2)) regular sparse column matrix")
    println(io, "Row indices: ", A.rowval)
    print(io, "Non-zero values: ", A.nzval)
end
