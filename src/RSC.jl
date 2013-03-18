## Regular sparse column-oriented matrices

## The representation is as two dense matrices the non-zero values and
## the row indices.  This requires that the number of nonzeros in each
## column be constant.  The row indices should be sorted within columns

## To allow for both random effects and fixed-effects terms in a mixed-effects
## model the non-zero values can have more rows than the row indices, the
## convention being that the p extra rows are dense rows appended to the matrix.
module RSC

using GLM, DataFrames
using Base.LinAlg.CHOLMOD.CholmodFactor
using Base.LinAlg.CHOLMOD.CholmodSparse
using Base.LinAlg.CHOLMOD.CholmodSparse!

export
   SparseMatrixRSC,
   RSCpred,
   update!

import Base.(*)
import Base.A_mul_Bc
import Base.A_mul_Bt
import Base.copy
import Base.dense
import Base.full
import Base.nnz
import Base.show
import Base.size
import Base.sparse

typealias VTypes Union(Float64,Complex128)
typealias ITypes Union(Int32,Int64)

type SparseMatrixRSC{Tv<:VTypes,Ti<:ITypes} <: AbstractSparseMatrix{Tv,Ti}
    q::Int                              # number of rows in the Zt part
    p::Int                              # number of rows in the Xt part
    rowval::Matrix{Ti}                  # row indices of nonzeros
    nzval::Matrix{Tv}                   # nonzero values
    function SparseMatrixRSC(rowval::Matrix{Ti},nzval::Matrix{Tv})
        if size(nzval,2) != size(rowval,2)
            error("number of columns in nzval, $(size(nzval,2)), should be $(size(rowval,2))")
        end
        if !all(rowval .> 0) error("row values must be positive") end
        new(int(max(rowval)), size(nzval,1) - size(rowval,1), rowval, nzval)
    end
end

function SparseMatrixRSC(rowval::Matrix,nzval::Matrix)
    SparseMatrixRSC{eltype(nzval),eltype(rowval)}(rowval,nzval)
end

function SparseMatrixRSC{Tv<:VTypes,Ti<:Integer}(rowval::Vector{Ti},
                                                 nzval::Matrix{Tv})
    SparseMatrixRSC(int32(rowval)',nzval)
end

function SparseMatrixRSC(rowval::PooledDataArray,nzval::Matrix)
    SparseMatrixRSC(int32(rowval.refs)',nzval)
end

function *{T}(A::SparseMatrixRSC{T}, v::Vector{T})
    m,n = size(A)
    if length(v) != n error("Dimension mismatch") end
    res = zeros(T, m)
    rv  = A.rowval
    nv  = A.nzval
    k   = size(rv,1)
    ## Sparse part
    for j in 1:n, i in 1:k
        res[rv[i,j]] += v[j] * nv[i,j]
    end
    ## Dense part
    for j in 1:n, i in 1:A.p
        res[A.q + i] += v[j] * nv[k + i, j]
    end
    res
end

function A_mul_Bc{Tv,Ti<:ITypes}(A::SparseMatrixRSC{Tv,Ti},
                                 B::SparseMatrixRSC{Tv,Ti})
    if is(A,B)
        chsp = CholmodSparse!(sparse(A),0)
        return chsp*chsp'
    end
    return CholmodSparse!(sparse(A),0) * CholmodSparse!(sparse(B),0)'
end

copy(A::SparseMatrixRSC) = SparseMatrixRSC(copy(A.rowval), copy(A.nzval))

function expandi{Tv,Ti}(A::SparseMatrixRSC{Tv,Ti})
    A.p == 0 ? vec(A.rowval) :
    vec(vcat(A.rowval, mapreduce(i->fill(convert(Ti,i+A.q), (1,size(A,2))),
                                 vcat, 1:A.p)))
end

function expandj{Tv,Ti}(A::SparseMatrixRSC{Tv,Ti})
    vec(diagmm(ones(Ti,size(A.nzval)), convert(Vector{Ti},[1:size(A,2)])))
end

function expandp{Tv,Ti}(A::SparseMatrixRSC{Tv,Ti})
    one(Ti) + convert(Vector{Ti}, [0:size(A.nzval,1):nnz(A)])
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

function sparse(x::SparseMatrixRSC)
    SparseMatrixCSC(size(x,1), size(x,2), expandp(x), expandi(x), vec(x.nzval))
end

## DyeStuff example from the lme4 package for R
# using DataFrames
# SparseMatrixRSC(gl(6,5), ones((2,30)))
#ZXtZX = chm_sort(chm_aat(ZXt))
#fac = chm_factorize(aat)
#Yield = float([1545, 1440, 1440, 1520, 1580, 1540, 1555, 1490, 1560, 1495, 1595, 1550, 1605, 1510, 1560, 1445, 1440, 1595, 1465, 1545, 1595, 1630, 1515, 1635, 1625, 1520, 1455, 1450, 1480, 1445])

function mktheta(nc)
    mapreduce(j->mapreduce(k->float([1.,zeros(j-k)]), vcat, 1:j), vcat, nc)
end

type RSCpred{Tv<:VTypes,Ti<:ITypes} <: LinPred
    ZXt::SparseMatrixRSC{Tv,Ti}
    theta::Vector{Tv}
    lower::Vector{Tv}
    A::CholmodSparse{Tv,Ti}
    L::CholmodFactor{Tv,Ti}
    ubeta0::Vector{Tv}
    delubeta::Vector{Tv}
end

function RSCpred{Tv,Ti}(ZXt::SparseMatrixRSC{Tv,Ti}, theta::Vector)
    aat = ZXt*ZXt'
    cp = aat.colptr0
    rv = aat.rowval0
    xv = aat.nzval
    for j in 1:ZXt.q                    # inflate the diagonal
        k = cp[j+1]                     # 0-based indices in cp
        assert(rv[k] == j-1)
        xv[k] += 1.
    end
    th  = convert(Vector{Tv},theta)
    ff  = sum(th .== one(Tv)) 
    if ff != size(ZXt.rowval, 1)
        error("number of finite elements of lower = $ff should be $(size(ZXt.rowval, 1))")
    end
    ub = zeros(Tv,(size(ZXt,1),))
    RSCpred{Tv,Ti}(ZXt, th, [convert(Tv,t == 0.?-Inf:0.) for t in th],
                   aat, cholfact(aat,true), ub, ub)
end

function apply_lambda!{T}(vv::Vector{T}, x::RSCpred{T}, wt::T)
    dpos = 0
    low  = x.lower
    th   = x.theta
    off  = 0
    for k in 1:length(low)
        if low[k] == 0.                 # diagonal element of factor
            dpos += 1
            vv[dpos] *= th[k]
            off = 0
        else
            off += 1
            vv[dpos] += th[k] * vv[dpos + off]
        end
    end
    if (wt != 1.) vv *= wt end
    vv
end
    
function update!{Tv,Ti}(x::RSCpred{Tv,Ti}, theta::Vector{Tv},
                        resid::Vector{Tv}, wts::Vector{Tv})
    if length(theta) != length(x.theta)
        error("length(theta) = $(length(theta)), should be $(length(x.theta))")
    end
    if any(theta .< x.lower) error("theta violates lower bound") end
    n = size(x.ZXt,2)
    if (length(resid) != n || length(wts) != n)
        error("length(resid) = $(length(resid)), length(wts) = $(length(wts)) should be $n")
    end
    x.theta[:] = theta                  # in-place install of new value of theta
    cp  = increment(x.A.colptr0)        # 1-based column pointers and rowvals for A 
    rv  = increment(x.A.rowval0)
    nzv = x.A.nzval
    q   = x.ZXt.q         # number of rows and columns in Zt part of A
    ## Initialize A to the q by q identity in the upper left hand corner,
    ## zeros elsewhere.
    for j in 1:(x.A.c.n), kk in cp[j]:(cp[j+1] - 1)
        nzv[kk] = (rv[kk] == j && j <= q) ? 1. : 0.
    end
    ZXr = x.ZXt.rowval
    ZXv = x.ZXt.nzval
    k   = size(ZXr, 1) # number of non-zeros per column of the Zt part
    w = Array(Tv, size(ZXv, 1))     # avoid reallocation of work array
    for j in 1:n
        w[:] = ZXv[:,j]
        apply_lambda!(w, x, wts[j])
        ## scan up the j'th column of ZXt
        for i in length(w):-1:1
            ii = i <= k ? ZXr[i,j] : q + i - k
            cpi = cp[ii]                # 1-based column pointer
            ll = cp[ii + 1] - 1         # location of diagonal
            nzv[ll] += square(w[i])
            for l in (i-1):-1:1         # off-diagonals
                if ll < cpi break end
                ii1 = l <= k ? ZXr[l,j] : q + l - k
                while (rv[ll] > ii1) ll -= 1 end
                if rv[ll] != ii1 error("Pattern mismatch") end
                nzv[ll] += w[i] * w[l]
            end
        end
    end
    chm_factorize!(x.L.c, x.A.c)
end

function update!{T}(x::RSCpred{T}, theta::Vector{T}, resid::Vector{T})
    update!(x, theta, resid, ones(T,length(resid)))
end

end
