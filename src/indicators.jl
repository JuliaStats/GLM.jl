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

## Sparse indicator matrix of one or more vectors (indicators are concatenated)
function spind{T}(x::AbstractVector{T})
    levs = unique(x, true)
    d = Dict{T, Int32}()
    for i in 1:length(levs) d[levs[i]] = i end
    sparse(int32([1:length(x)]), ([d[el]::Int32 for el in x]), 1.), levs
end

function spind{T}(x::AbstractVector{T}...)
    mm = map(spind, x)
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
