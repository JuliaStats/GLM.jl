abstract Link             # Link types define linkfun!, linkinv!, and mueta!

@Base.math_const sqrt2 1.4142135623730951 sqrt(big(2))
@Base.math_const sqrt2pi 2.5066282746310007 sqrt(big(2)*pi)

type CauchitLink <: Link end
type CloglogLink  <: Link end
type IdentityLink <: Link end
type InverseLink  <: Link end
type LogitLink <: Link end
type LogLink <: Link end
type ProbitLink <: Link end

type CauchLink <: UnaryFunctor end
evaluate{T<:FloatingPoint}(::CauchLink,x::T) = tan(pi*(x - convert(T,0.5)))
type CauchInv <: UnaryFunctor end
evaluate{T<:FloatingPoint}(::CauchInv,x::T) = convert(T,0.5) + atan(x)/pi
type CauchME <: UnaryFunctor end
evaluate{T<:FloatingPoint}(::CauchME,x::T) = one(T)/(pi*(one(T) + abs2(x)))
type CLgLgLink <: UnaryFunctor end
evaluate{T<:FloatingPoint}(::CLgLgLink,x::T) = log(-log(one(T) - x))
type CLgLgInv <: UnaryFunctor end
evaluate{T<:FloatingPoint}(::CLgLgInv,x::T) = -expm1(-exp(x))
type CLgLgME <: UnaryFunctor end
evaluate{T<:FloatingPoint}(::CLgLgME,x::T) = exp(x)*exp(-exp(x))
type InvME <: UnaryFunctor end
evaluate{T<:FloatingPoint}(::InvME,x::T) = -one(T)/abs2(x)
type LogitME <: UnaryFunctor end
evaluate{T<:FloatingPoint}(::LogitME,x::T) = (e = exp(-abs(x)); f = one(T) + e; e / (f * f))
type ProbLink <: UnaryFunctor end
evaluate{T<:FloatingPoint}(::ProbLink,x::T) = sqrt2*erfinv(convert(T,2.0)*x-one(T))
type ProbInv <: UnaryFunctor end
evaluate{T<:FloatingPoint}(::ProbInv,x::T) = (one(T) + erf(x/sqrt2))/convert(T,2.0)
type ProbME <: UnaryFunctor end
evaluate{T<:FloatingPoint}(::ProbME,x::T) = exp(-x*x/convert(T,2.0))/sqrt2pi

linkfun!{T<:FloatingPoint}(::CauchitLink, eta::Vector{T}, mu::Vector{T}) = map!(CauchLink(),eta,mu)
linkinv!{T<:FloatingPoint}(::CauchitLink, mu::Vector{T}, eta::Vector{T}) = map!(CauchInv(),mu,eta)
mueta!{T<:FloatingPoint}(::CauchitLink, me::Vector{T}, eta::Vector{T}) = map!(CauchME(),me,eta)

linkfun!{T<:FloatingPoint}(::CloglogLink, eta::Vector{T}, mu::Vector{T}) = map!(CLgLgLink(),eta,mu)
linkinv!{T<:FloatingPoint}(::CloglogLink, mu::Vector{T}, eta::Vector{T}) = map!(CLgLgInv(),mu,eta)
mueta!{T<:FloatingPoint}(::CloglogLink, me::Vector{T}, eta::Vector{T}) = map!(CLgLgME(),me,eta)

linkfun!{T<:FloatingPoint}(::IdentityLink, eta::Vector{T}, mu::Vector{T}) = copy!(eta,mu)
linkinv!{T<:FloatingPoint}(::IdentityLink, mu::Vector{T}, eta::Vector{T}) = copy!(mu,eta)         
mueta!{T<:FloatingPoint}(::IdentityLink, me::Vector{T}, eta::Vector{T}) = fill!(me,one(T))

linkfun!{T<:FloatingPoint}(::InverseLink, eta::Vector{T}, mu::Vector{T}) = map!(Divide(),eta,1.,mu)
linkinv!{T<:FloatingPoint}(::InverseLink, mu::Vector{T}, eta::Vector{T}) = map!(Divide(),mu,1.,eta)
mueta!{T<:FloatingPoint}(::InverseLink, me::Vector{T}, eta::Vector{T}) = map!(InvME(),me,eta)

linkfun!{T<:FloatingPoint}(::LogitLink, eta::Vector{T}, mu::Vector{T}) = map!(Logit(),eta,mu)
linkinv!{T<:FloatingPoint}(::LogitLink, mu::Vector{T}, eta::Vector{T}) = map!(NumericExtensions.Logistic(),mu,eta)
mueta!{T<:FloatingPoint}(::LogitLink, me::Vector{T}, eta::Vector{T}) = map!(LogitME(),me,eta)

linkfun!{T<:FloatingPoint}(::LogLink, eta::Vector{T}, mu::Vector{T}) = map!(Log(),eta,mu)
linkinv!{T<:FloatingPoint}(::LogLink, mu::Vector{T}, eta::Vector{T}) = map!(Exp(),mu,eta)
mueta!{T<:FloatingPoint}(::LogLink, mu::Vector{T}, eta::Vector{T}) = map!(Exp(),mu,eta)

linkfun!{T<:FloatingPoint}(::ProbitLink, eta::Vector{T}, mu::Vector{T}) = map!(ProbLink(),eta,mu)
linkinv!{T<:FloatingPoint}(::ProbitLink, mu::Vector{T}, eta::Vector{T}) = map!(ProbInv(),mu,eta)
mueta!{T<:FloatingPoint}(::ProbitLink, me::Vector{T}, eta::Vector{T}) =  map!(ProbME(),me,eta)

canonicallink(::Gamma) = InverseLink()
canonicallink(::Normal) = IdentityLink()
canonicallink(::Bernoulli) = LogitLink()
canonicallink(d::Poisson) = LogLink()

type BernoulliVar <: UnaryFunctor end
evaluate{T<:FloatingPoint}(::BernoulliVar,x::T) = x * (one(T) - x)

var!{T<:FloatingPoint}(d::Bernoulli,v::Vector{T},mu::Vector{T}) = map!(BernoulliVar(),v,mu)
var!{T<:FloatingPoint}(d::Gamma,v::Vector{T},mu::Vector{T}) = map!(Abs2(),v,mu)
var!{T<:FloatingPoint}(d::Normal,v::Vector{T},mu::Vector{T}) = fill!(v,one(T))
var!{T<:FloatingPoint}(d::Poisson,v::Vector{T},mu::Vector{T}) = copy!(v,mu)

type BernoulliMuStart <: BinaryFunctor end
evaluate{T<:FloatingPoint}(::BernoulliMuStart,y::T,w::T) = (w*y + convert(T,0.5))/(w + one(T))
result_type{T<:FloatingPoint}(::BernoulliMuStart,::Type{T},::Type{T}) = T

mustart{T<:FloatingPoint}(d::Bernoulli,y::Vector{T},wt::Vector{T}) = map(BernoulliMuStart(),y,wt)
mustart{T<:FloatingPoint}(d::Gamma,y::Vector{T},wt::Vector{T}) = copy(y)
mustart{T<:FloatingPoint}(d::Normal,y::Vector{T},wt::Vector{T}) = copy(y)
mustart{T<:FloatingPoint}(d::Poisson,y::Vector{T},wt::Vector{T}) = map(Add(), y, convert(T,0.1))

for Op in [:BernoulliVar,
           :CauchLink, :CauchInv, :CauchME,
           :CLgLgLink, :CLgLgInv, :CLgLgME,
           :InvME,
           :LogitME,
           :ProbLink, :ProbInv, :ProbME]
    @eval result_type{T<:FloatingPoint}(::$(Op), ::Type{T}) = T
end


type BernoulliDevResid <: TernaryFunctor end
function evaluate{T<:FloatingPoint}(::BernoulliDevResid,y::T,mu::T,wt::T)
    omy = one(T)-y
    2.wt*(xlogy(y,y/mu) + xlogy(omy,omy/(one(T)-mu)))
end
result_type{T<:FloatingPoint}(::BernoulliDevResid,::Type{T},::Type{T},::Type{T}) = T

type PoissonDevResid <: TernaryFunctor end
evaluate{T<:FloatingPoint}(::PoissonDevResid,y::T,mu::T,wt::T) = 2.wt * (xlogy(y,y/mu) - (y - mu))
result_type{T<:FloatingPoint}(::PoissonDevResid,::Type{T},::Type{T},::Type{T}) = T

function devresid!{T<:FloatingPoint}(d::Bernoulli,dr::Vector{T},y::Vector{T},
                                     mu::Vector{T},wt::Vector{T})
    map!(BernoulliDevResid(), dr, y, mu, wt)
end
function devresid!{T<:FloatingPoint}(d::Poisson,dr::Vector{T},y::Vector{T},
                                     mu::Vector{T},wt::Vector{T})
    map!(PoissonDevResid(), dr, y, mu, wt)
end
