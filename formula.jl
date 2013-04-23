## Utilities to work with the random-effects part of the formula

## Extract the random effects terms as a vector of expressions
function retrms(mf::ModelFrame)
    convert(Vector{Expr}, filter(x->Meta.isexpr(x,:call) && x.args[1] == :|, mf.terms.terms))
end

## Check if all random-effects terms are simple
issimple(terms::Vector{Expr}) = all(map(x->x.args[2] == 1, terms))

## Return the grouping factors as a matrix - convert to int32 if feasible
function grpfac(terms::Vector{Expr}, mf::ModelFrame)
    m = hcat(map(x->mf.df[x.args[3]].refs,terms)...)
    q = sum([max(m[:,j]) for j in 1:size(m,2)])
    q < typemax(Int32) ? int32(m) : int(m)
end

const template = Formula(:(~ foo))

function lhs(trms::Vector{Expr}, mf::ModelFrame)
    map(x->(template.rhs = x.args[2]; ModelMatrix(ModelFrame(template, mf.df))), trms)
end

## extract the data values as vectors from various compound types
dv(v::DataVector) = v.data
dv(v::Vector) = v
dv(v::PooledDataVector) = v.refs

