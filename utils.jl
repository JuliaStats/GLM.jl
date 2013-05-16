## Utilities to work with the random-effects part of the formula

## Extract the random effects terms as a vector of expressions
function retrms(mf::ModelFrame)
    convert(Vector{Expr}, filter(x->Meta.isexpr(x,:call) && x.args[1] == :|, mf.terms.terms))
end

## Check if all random-effects terms are simple
issimple(terms::Vector{Expr}) = all(map(issimple, terms))
issimple(expr::Expr) = Meta.isexpr(expr,:call) && expr.args[1] == :| && expr.args[2] == 1
## Return the grouping factors as a matrix
function grpfac(e::Expr, mf::ModelFrame)
    if !Meta.isexpr(e,:call) || e.args[1] != :|
        error("Expression $e is not a random-effects term")
    end
    mf.df[e.args[3]].refs
end
function grpfac(terms::Vector{Expr}, mf::ModelFrame)
    hcat(map(x->grpfac(x,mf),terms)...)
end

const template = Formula(:(~ foo))

function lhs(trms::Vector{Expr}, mf::ModelFrame)
    map(x->lhs(x, mf), trms)
end
function lhs(expr::Expr, mf::ModelFrame)
    if !(Meta.isexpr(expr,:call) && expr.args[1] == :|)
        error("expr = $expr and should be a call to the ':|' function")
    end
    if expr.args[2] == 1 return ModelMatrix(ones(size(mf.df,1), 1), [0]) end
    template.rhs = expr.args[2]
    ModelMatrix(ModelFrame(template, mf.df))
end    

## extract the data values as vectors from various compound types
dv(v::DataVector) = v.data
dv(v::Vector) = v
dv(v::PooledDataVector) = v.refs

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
