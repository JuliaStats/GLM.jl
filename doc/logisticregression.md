# Large-scale logistic regression in [Julia](http://julialang.org)

A
[recent thread](https://groups.google.com/forum/#!topic/julia-users/Izq1DjfMhek)
on the `julia-users` discussion group concerned computational methods
for
[logistic regression](http://en.wikipedia.org/wiki/Logistic_regression)
and how well they scale to large problems.  The timing is fortuitous
given the recent release of Dahua Lin's
[NumericFunctors](https://github.com/lindahua/NumericFunctors.jl)
package for Julia.  It turns out that logistic regression provides a
great example of the benefit of the techniques in this package.

## Model formulation

The
[Wikipedia article](http://en.wikipedia.org/wiki/Logistic_regression)
describes the logistic regression model for an _n_-dimensional binary
response vector, _y_, in terms of a model matrix _X_ of size _(n,p)_
and a _p_-dimensional coefficient vector, _beta_.  The *linear
predictor* vector, _eta = X*beta_, and the vector of mean responses,
_mu_, which in this case are the probabilities of a positive response
in a Bernoulli distribution, are componentwise transformations of each
other.  The scalar function taking an element of _mu_ to the
corresponding element of _eta_ is called the *link function*.  The
*canonical link function* for the Bernoulli distribution is the
*log-odds* or *logit* function
```julia
logit(mu::Real) = log(mu/(1.-mu))
```
The inverse link function
```julia
logistic(eta::Real) = 1./(1.+exp(-eta))
```
is called the *logistic* function.  Determining the *maximum
likelihood* estimates of the coefficients _beta_ given the observed
binary responses, _y_, and the model matrix, _X_, is a form of
*logistic regression*.

## Using functors

Dahua Lin's NumericFunctors package provides incredibly efficient
methods for transforming, say, one numeric vector to another via a
scalar function, which is exactly what the link and the inverse link
functions do.  Creating a functor is only a bit more complicated than
writing a function definition.  The functor is a type for which
methods for `evaluate` and `result_type` are defined.

```julia
using NumericFunctors
import NumericFunctors: evaluate, result_type

type Logit <: UnaryFunctor end
evaluate(::Logit, mu) = log(mu/(1.-mu))
result_type(::Logit,t::Type) = NumericFunctors.to_fptype(t)

type Logistic <: UnaryFunctor end
evaluate(::Logistic, eta) = 1./(1.+exp(-eta))
result_type(::Logistic,t::Type) = NumericFunctors.to_fptype(t)
```
By convention functor names are capitalized as they are names of
types.  Function names are usually in lower case.

As described in the documentation for the NumericFunctors package, 
functors can be used with functions such as `map`, `map!` and `map!`
to update one vector from other vectors.  An advantage of this
approach is that the update can be applied in place.  Let us define a
type

```julia
type LogisticRegression{T<:Float64}
    X::Matrix{T}
	y::BitArray{1}
	beta::Vector{T}
	mu::Vector{T}
	eta::Vector{T}
	function LogisticRegression(X::Matrix,y::Vector)
	    Xm = float(X); yy = convert(BitArray{1},y)
		n,p = size(Xm); length(yy) == n || error("Dimension mismatch")
		mu = Float64[v ? 0.75 : 0.25 for v in yy]
		new(Xm,yy,zeros(p),mu,map(Logit,mu))
    end
end
```
