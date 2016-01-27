using Base.Cartesian

abstract Link             # Link types define linkfun!, linkinv!, and mueta!

type CauchitLink <: Link end
type CloglogLink  <: Link end
type IdentityLink <: Link end
type InverseLink  <: Link end
type LogitLink <: Link end
type LogLink <: Link end
type ProbitLink <: Link end
type SqrtLink <: Link end

"""
Link function mapping mu to eta, the linear predictor (CauchitLink response model).
"""
linkfun(::CauchitLink, μ) = tan(pi*(μ - oftype(μ, 0.5)))
"""
Inverse Link function mapping mu to eta, the linear predictor (CauchitLink response model).
"""
linkinv(::CauchitLink, η) = oftype(η, 0.5) + atan(η)/pi
mueta(::CauchitLink, η) = one(η)/(pi*(one(η) + abs2(η)))

"""
Link function mapping mu to eta, the linear predictor (CloglogLink response model).
"""
linkfun(::CloglogLink, μ) = log(-log1p(-μ))
"""
Inverse Link function mapping mu to eta, the linear predictor (CloglogLink response model).
"""
linkinv(::CloglogLink, η) = -expm1(-exp(η))
"""
Derivative of inverse link.
"""
mueta(::CloglogLink, η) = exp(η)*exp(-exp(η))

"""
Link function mapping mu to eta, the linear predictor (IdentityLink response model).
"""
linkfun(::IdentityLink, μ) = μ
"""
Inverse Link function mapping mu to eta, the linear predictor (IdentityLink response model).
"""
linkinv(::IdentityLink, η) = η
"""
Derivative of inverse link.
"""
mueta(::IdentityLink, η) = 1

"""
Link function mapping mu to eta, the linear predictor (InverseLink response model).
"""
linkfun(::InverseLink, μ) = 1/μ
"""
Inverse Link function mapping mu to eta, the linear predictor (InverseLink response model).
"""
linkinv(::InverseLink, η) = 1/η
"""
Derivative of inverse link.
"""
mueta(::InverseLink, η) = -inv(abs2(η))

"""
Link function mapping mu to eta, the linear predictor (LogitLink response model).
"""
linkfun(::LogitLink, μ) = logit(μ)
"""
Inverse mapping mu to eta, the linear predictor (LogitLink response model).
"""
linkinv(::LogitLink, η) = logistic(η)
"""
Derivative of inverse link.
"""
mueta(::LogitLink, η) = (e = exp(-abs(η)); f = one(η) + e; e / (f * f))

"""
Link function mapping mu to eta, the linear predictor (LogLink response model).
"""
linkfun(::LogLink, μ) = log(μ)
"""
Inverse Link function mapping mu to eta, the linear predictor (LogLink response model).
"""
linkinv(::LogLink, η) = exp(η)
"""
Derivative of inverse link.
"""
mueta(::LogLink, η) = exp(η)

"""
Link function mapping mu to eta, the linear predictor (ProbitLink response model).
"""
linkfun(::ProbitLink, μ) = -sqrt2*erfcinv(2*μ)
"""
Inverse Link function mapping mu to eta, the linear predictor (ProbitLink response model).
"""
linkinv(::ProbitLink, η) = erfc(-η/sqrt2)/2
"""
Derivative of inverse link.
"""
mueta(::ProbitLink, η) = exp(-η^2/2)/sqrt2π

"""
Link function mapping mu to eta, the linear predictor (SqrtLink responce model). 
"""
linkfun(::SqrtLink, μ) = sqrt(μ)
"""
Inverse Link function mapping mu to eta, the linear predictor (SqrtLink responce model). 
"""
linkinv(::SqrtLink, η) = abs2(η)
"""
Derivative of inverse link.
"""
mueta(::SqrtLink, η) = 2η

"""
Canonical link function for a Binomial distribution.
"""
canonicallink(::Binomial) = LogitLink()
"""
Canonical link function for a Gamma distribution.
"""
canonicallink(::Gamma) = InverseLink()
"""
Canonical link function for a Normal distribution.
"""
canonicallink(::Normal) = IdentityLink()
"""
Canonical link function for a Poisson distribution.
"""
canonicallink(::Poisson) = LogLink()

# For the "odd" link functions we evaluate the linear predictor such that mu is closest to zero where the precision is higher
function glmvar(::Binomial, link::@compat(Union{CauchitLink,InverseLink,LogitLink,ProbitLink}), μ, η)
    μ = linkinv(link, ifelse(η < 0, η, -η))
    μ*(1-μ)
end
glmvar(::Binomial, ::Link, μ, η) = μ*(1-μ)
glmvar(::Gamma, ::Link, μ, η) = abs2(μ)
glmvar(::Normal, ::Link, μ, η) = 1
glmvar(::Poisson, ::Link, μ, η) = μ

"""
Derive starting values for the mu vector.
"""
mustart(::Binomial, y, wt) = (wt*y + oftype(y,0.5))/(wt + one(y))
"""
Derive starting values for the mu vector.
"""
mustart(::Gamma, y, wt) = y
"""
Derive starting values for the mu vector.
"""
mustart(::Normal, y, wt) = y
"""
Derive starting values for the mu vector.
"""
mustart(::Poisson, y, wt) = y + oftype(y, 0.1)

"""
Return a vector of squared deviance residuals for Binomial distributed input data.
"""
function devresid(::Binomial, y, μ, wt)
    if y == 1
        return 2.0*wt*-log(μ)
    elseif y == 0
        return -2.0*wt*log1p(-μ)
    else
        return 2.0*wt*(y*(log(y) - log(μ)) + (1 - y)*(log1p(-y) - log1p(-μ)))
    end
end
"""
Return a vector of squared deviance residuals for Gamma distributed input data.
"""
devresid(::Gamma, y, μ, wt) = -2wt * (log(y/μ) - (y - μ)/μ)
"""
Return a vector of squared deviance residuals for Normal distributed input data.
"""
devresid(::Normal, y, μ, wt) = wt * abs2(y - μ)
"""
Return a vector of squared deviance residuals for Poisson distributed input data.
"""
devresid(::Poisson, y, μ, wt) = 2wt * (xlogy(y,y/μ) - (y - μ))
