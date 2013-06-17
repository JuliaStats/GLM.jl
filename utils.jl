## Utilities to work with the random-effects part of the formula


## Check if all random-effects terms are simple
issimple(terms::Vector{Expr}) = all(map(issimple, terms))
issimple(expr::Expr) = Meta.isexpr(expr,:call) && expr.args[1] == :| && expr.args[2] == 1

## Add an identity block along inds to a symmetric A stored in the upper triangle
function pluseye!{T}(A::CholmodSparse{T}, inds)
    if A.c.stype <= 0 error("Matrix A must be symmetric and stored in upper triangle") end
    cp = A.colptr0
    rv = A.rowval0
    xv = A.nzval
    for j in inds
        k = cp[j+1]
        assert(rv[k] == j-1)
        xv[k] += one(T)
    end
    A
end
pluseye!(A::CholmodSparse) = pluseye!(A,1:size(A,1))

function fill!{T}(a::Vector{T}, x::T, inds)
    for i in inds a[i] = x end
end

# fill in the lower triangle of a k by k matrix with the vector th
function mkLambda(k::Integer,th::Vector{Float64})
    if length(th) != (k*(k+1))>>1
        error("length(th) = $(length(th)) should be $((k*(k+1))>>1) for k = $k")
    end
    tt = zeros(k*k); tt[bool(vec(tril(ones(k,k))))] = th
    reshape(tt, (k, k))
end
