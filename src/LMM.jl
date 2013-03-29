module LMM                              # linear mixed models

using RSC, NLopt
using Base.LinAlg.CHOLMOD.CholmodFactor
using Base.LinAlg.CHOLMOD.CholmodSparse
using Base.LinAlg.CHOLMOD.chm_scale!
using Base.LinAlg.CHOLMOD.chm_factorize!

export LMMsimple
import Distributions.deviance
import Distributions.fit
import Base.fill!

typealias Inds Union(Int, Vector{Int}, Range{Int}, Range1{Int})
typealias ITypes Union(Int32,Int64)

type LMMsimple{Tv<:Float64,Ti<:ITypes}
    ZXt::SparseMatrixRSC{Tv,Ti}# model matrices Z and X in an RSC structure
    theta::Vector{Tv}          # variance component parameter vector
    lower::Vector{Tv}          # lower bounds (always zeros(length(theta)) for these models)
    A::CholmodSparse{Tv,Ti}    # sparse symmetric system matrix
    anzv::Vector{Tv}           # cached copy of nonzeros from ZXt*ZXt'
    L::CholmodFactor{Tv,Ti}    # factor of current system matrix
    ubeta::Vector{Tv}          # coefficient vector
    sqrtwts::Vector{Tv}        # square root of weights - can be length 0
    y::Vector{Tv}              # response vector
    ZXty::Vector{Tv}           # cached copy of ZXt*y
    lambda::Vector{Tv}         # diagonal of Lambda
    mu::Vector{Tv}             # fitted response vector
end

## Add an identity block on Inds to a symmetric A stored in the upper triangle 
function pluseye{T}(A::CholmodSparse{T}, inds::Inds) 
    if A.c.stype <= 0 error("Matrix A must be symmetric and stored in upper triangle") end
    cp = A.colptr0
    rv = A.rowval0
    xv = A.nzval
    for j in inds
        k = cp[j+1]
        assert(rv[k] == j-1)
        xv[k] += one(T)
    end
end

pluseye(A::CholmodSparse) = pluseye(A,1:size(A,1))

function LMMsimple{Tv<:Float64,Ti<:ITypes}(ZXt::SparseMatrixRSC{Tv,Ti},
                                           y::Vector{Tv},wts::Vector{Tv})
    n = size(ZXt,2)
    if length(y) != n error("size(ZXt,2) = $(size(ZXt,2)) and length(y) = $length(y)") end
    lwts = length(wts)
    if lwts != 0 && lwts != n error("length(wts) = $lwts should be 0 or length(y) = $n") end
    A = A_mul_Bc(ZXt,ZXt) # i.e. ZXt*ZXt' (single quotes confuse syntax highlighting)
    anzv = copy(A.nzval)
    pluseye(A, 1:ZXt.q)
    L = cholfact(A)
    ZXty = ZXt*y
    ubeta = vec((L\ZXty).mat)
    k = size(ZXt.rowval, 1)
    LMMsimple{Tv,Ti}(ZXt, ones(k), zeros(k), A, anzv, L,
                     ubeta, sqrt(wts), y, ZXty, ones(size(L,1)), Ac_mul_B(ZXt,ubeta))
end

LMMsimple(ZXt,y) = LMMsimple(ZXt, y, Array(eltype(y), 0))

function LMMsimple{Tv<:Float64,Ti<:ITypes}(inds::Matrix{Ti},
                                           X::Matrix{Tv},
                                           y::Vector{Tv},
                                           wts::Vector{Tv})
    n = length(y)
    if !(size(inds,1) == size(X,1) == n) error("Dimension mismatch") end
    ii = copy(inds)
    for j in 2:size(ii,2) ii[:,j] += max(ii[:,j-1]) end
    LMMsimple(SparseMatrixRSC(ii', [ones(size(ii)) X]'), y, wts)
end

function LMMsimple{Tv<:Float64,Ti<:ITypes}(inds::Matrix{Ti}, X::Matrix{Tv}, y::Vector{Tv})
    LMMsimple(inds, X, y, Array(Tv, 0))
end

function fill!{T,Ti<:Inds}(a::Vector{T}, x::T, inds::Ti)
    for i in inds a[i] = x end
end

function deviance(m::LMMsimple,theta::Vector{Float64}, REML::Bool)
    if length(theta) != length(m.theta) error("Dimension mismatch") end
    if any(theta .< 0.) error("all elements of theta must be non-negative") end
    m.theta[:] = theta[:]               # copy in place
    m.A.nzval[:] = m.anzv[:]            # restore A in place to ZXt*ZXt'
    ZXt = m.ZXt
    q = ZXt.q
    p = ZXt.p
    for i in 1:length(theta) fill!(m.lambda, theta[i], int(ZXt.rowrange[i])) end
    chm_scale!(m.A, m.lambda, 3) # symmetric scaling
    pluseye(m.A, 1:q)
    chm_factorize!(m.L,m.A)
    m.ubeta[:] = vec((m.L \ (m.lambda .* m.ZXty)).mat)[:]
    m.mu[:] = Ac_mul_B(ZXt, m.lambda .* m.ubeta)
    rss = (s = 0.; for r in (m.y - m.mu) s += r*r end; s)
    ussq = (s = 0.; for j in 1:q s += square(m.ubeta[j]) end; s)
    lnum = log(2pi * (rss + ussq))
    if REML
        nmp = float(size(ZXt,2) - p)
        return logdet(m.L) + nmp * (1 + lnum - log(nmp))
    end
    n = float(size(ZXt,2))
    return logdet(m.L, 1:q) + n * (1 + lnum - log(n))
end

deviance(m::LMMsimple,theta::Vector{Float64}) = deviance(m, theta, false)
# Dyestuff example
# ZXt = SparseMatrixRSC(1+div([0:29],5)), ones((2,30))
# Yield = [1545., 1440, 1440, 1520, 1580, 1540, 1555, 1490, 1560,
#                 1495, 1595, 1550, 1605, 1510, 1560, 1445, 1440, 1595,
#                 1465, 1545, 1595, 1630, 1515, 1635, 1625, 1520, 1455,
#                 1450, 1480, 1445]
# mm = LMMsimple(ZXt)

function fit(mm::LMMsimple, REML::Bool, verbose::Bool) # keyword arguments will help
    k = length(mm.theta)
    opt = Opt(:LN_BOBYQA, k)
    ftol_abs!(opt, 1e-6)    # criterion on deviance changes
    xtol_abs!(opt, 1e-6)    # criterion on all parameter value changes
    lower_bounds!(opt, mm.lower)
    function obj(x::Vector{Float64}, g::Vector{Float64})
        if length(g) > 0 error("gradient evaluations are not provided") end
        deviance(mm, x, REML)
    end
    if verbose
        count = 0
        function vobj(x::Vector{Float64}, g::Vector{Float64})
            count += 1
            val = obj(x, g)
            println("f_$count: $val, $x")
            val
        end
        min_objective!(opt, vobj)
    else
        min_objective!(opt, obj)
    end
    optimize(opt, mm.theta)
end

fit(mm::LMMsimple, REML::Bool) = fit(mm, REML, false)
fit(mm::LMMsimple) = fit(mm, false, false)

end

