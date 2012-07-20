require("linalg_suitesparse.jl")

# Generate levels - see the R documentation for gl
function gl(n::Integer, k::Integer, l::Integer)
    nk = n * k
    if l % nk != 0 error("length out must be a multiple of n * k") end
    aa = Array(Int, l)
    for j = 0:(l/nk - 1), i = 1:n
        aa[j * nk + (i - 1) * k + (1:k)] = i
    end
    aa
end

gl(n::Integer, k::Integer) = gl(n, k, n*k)

# Determine the unique values in an array
function unique{T}(x::AbstractArray{T}, sorted::Bool)
    d = Dict{T,Bool}()
    for el in x d[el] = true end
    sorted ? sort(keys(d)) : keys(d)
end

unique{T}(x::AbstractArray{T}) = unique(x, false)

# A cross-tabulation type.  Probably not a good design.
# Actually, this is just a one-way table
type xtab{T}
    vals::Array{T}
    counts::Array{Int, 1}
end

function xtab{T}(x::AbstractArray{T})
    d = Dict{T, Int}()
    for el in x d[el] = has(d, el) ? d[el] + 1 : 1 end
    kk = sort(keys(d))
    cc = Array(Int, numel(kk))
    for i in 1:numel(kk) cc[i] = d[kk[i]] end
    xtab(kk, cc)
end

# Another cross-tabulation function, this one leaves the result as a Dict
# Again, this is currently just for one-way tables.
function xtabs{T}(x::AbstractArray{T})
    d = Dict{T, length(x) > typemax(Int32) ? Int : Int32}()
    for el in x d[el] = has(d, el) ? d[el] + 1 : 1 end
    d
end

## dense or sparse matrix of indicators of the levels of a vector
function indicators{T}(sparseX::Bool, x::AbstractVector{T})
    levs = unique(x, true)
    nx   = length(x)
    nlev = length(levs)
    d    = Dict{T, Int}()
    for i in 1:nlev d[levs[i]] = i end
    ii   = 1:nx
    jj   = [d[el] for el in x]
    if sparseX return sparse(int32(ii), int32(jj), 1.), levs end
    X    = zeros(nx, nlev)
    for i in ii X[i, jj[i]] = 1. end
    X, levs
end

## default is dense indicators
indicators{T}(x::AbstractVector{T}) = indicators(x, false)

## indicators of multiple vectors
function indicators{T}(x::AbstractVector{T}...)
    mm = map(indicators, x)
    reduce(hcat, map(x->x[1], mm)), map(x->x[2], mm)
end

function indicators{T}(sparseX::Bool, x::AbstractVector{T}...)
    mm = map(v -> indicators(v, sparseX), x)
    reduce(hcat, map(x->x[1], mm)), map(x->x[2], mm)
end

function contr_treatment(n::Int, base::Int, contrasts::Bool, sparse::Bool)
    contr = sparse ? speye(n) : eye(n)
    if !contrasts return contr end
    if n < 2
        error(sprintf("contrasts not defined for %d degrees of freedom", n - 1))
    end
    contr[:, [1:(base-1), (base+1):n]]
end

contr_treatment(n::Int, base::Int, contrasts::Bool) = contr_treatment(n, base, contrasts, false)
contr_treatment(n::Int, base::Int) = contr_treatment(n, base, true, false)
contr_treatment(n::Int) = contr_treatment(n, 1, true, false)
