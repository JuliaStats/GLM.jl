## Additional methods for Distribution types.
## These methods differ from methods in base/distribution.jl in that
## the mean, mu, is passed as a vector or Real to these methods.  The
## distribution object d is used only for its type.

## FIXME: move this to base/distributions.jl 
function insupport{T<:Real}(d::Distribution, x::AbstractArray{T})
    for e in x
        if !insupport(d, e)
            return false
        end
    end
    true
end

## Distributions

canonicalLink(d::Gamma)     = InverseLink()
canonicalLink(d::Normal)    = IdentityLink()

canonicalLink(d::Bernoulli) = LogitLink()
logpmf( d::Bernoulli, mu::Real, y::Real) = y==0? log(1. - mu): (y==1? log(mu): -Inf)
mustart(d::Bernoulli,  y::Real, wt::Real) = (wt * y + 0.5)/(wt + 1)
var(    d::Bernoulli, mu::Real) = max(eps(), mu*(1. - mu))

canonicalLink(d::Poisson)   = LogLink()
logpmf(  d::Poisson, mu::Real, y::Real) = ccall(dlsym(_jl_libRmath,:dpois),Float64,(Float64,Float64,Int32),y,mu,1)
devResid(d::Poisson,  y::Real, mu::Real, wt::Real) = 2wt*((y==0? 0.: log(y/mu)) - (y-mu))
mustart( d::Poisson,  y::Real, wt::Real) = y + 0.1
var(     d::Poisson, mu::Real) = mu

## General definition of the (squared) deviance residuals
devResid(d::DiscreteDistribution, y::Real, mu::Real, wt::Real) = -2wt*logpmf(d, mu, y)
devResid(d::ContinuousDistribution, y::Real, mu::Real, wt::Real) = -2wt*logpdf(d, mu, y)

## Vectorized methods for distributions
function deviance{M<:Real,Y<:Real,W<:Real}(d::DiscreteDistribution,
                                           mu::AbstractArray{M},
                                           y::AbstractArray{Y},
                                           wt::AbstractArray{W})
    promote_shape(size(mu), promote_shape(size(y), size(wt))) # check for compatible sizes
    ans = 0.
    for i in 1:numel(y)
        ans += wt[i] * logpmf(d, mu[i], y[i])
    end
    -2ans
end
function deviance{M<:Real,Y<:Real,W<:Real}(d::ContinuousDistribution,
                                           mu::AbstractArray{M},
                                           y::AbstractArray{Y},
                                           wt::AbstractArray{W})
    promote_shape(size(mu), promote_shape(size(y), size(wt))) # check for compatible sizes
    ans = 0.
    for i in 1:numel(y)
        ans += wt[i] * logpdf(d, mu[i], y[i])
    end
    -2ans
end
function devResid{Y<:Real,M<:Real,W<:Real}(d::Distribution,
                                           y::AbstractArray{Y},
                                           mu::AbstractArray{M},
                                           wt::AbstractArray{W})
    R = Array(Float64, promote_shape(size(y), promote_shape(size(mu), size(wt))))
    for i in 1:numel(mu)
        R[i] = devResid(d, y[i], mu[i], wt[i])
    end
    R
end
function mustart{Y<:Real,W<:Real}(d::Distribution,
                                  y::AbstractArray{Y},
                                  wt::AbstractArray{W})
    M = Array(Float64, promote_shape(size(y), size(wt)))
    for i in 1:numel(M)
        M[i] = mustart(d, y[i], wt[i])
    end
    M
end
function var{M<:Real}(d::Distribution, mu::AbstractArray{M})
    V = similar(mu, Float64)
    for i in 1:numel(mu)
        V[i] = var(d, mu[i])
    end
    V
end

