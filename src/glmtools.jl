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
evaluate{T<:FP}(::CauchLink,x::T) = tan(pi*(x - convert(T,0.5)))
type CauchInv <: UnaryFunctor end
evaluate{T<:FP}(::CauchInv,x::T) = convert(T,0.5) + atan(x)/pi
type CauchME <: UnaryFunctor end
evaluate{T<:FP}(::CauchME,x::T) = one(T)/(pi*(one(T) + abs2(x)))
type CLgLgLink <: UnaryFunctor end
evaluate{T<:FP}(::CLgLgLink,x::T) = log(-log(one(T) - x))
type CLgLgInv <: UnaryFunctor end
evaluate{T<:FP}(::CLgLgInv,x::T) = -expm1(-exp(x))
type CLgLgME <: UnaryFunctor end
evaluate{T<:FP}(::CLgLgME,x::T) = exp(x)*exp(-exp(x))
type InvME <: UnaryFunctor end
evaluate{T<:FP}(::InvME,x::T) = -one(T)/abs2(x)
type LogitME <: UnaryFunctor end
evaluate{T<:FP}(::LogitME,x::T) = (e = exp(-abs(x)); f = one(T) + e; e / (f * f))
type ProbLink <: UnaryFunctor end
evaluate{T<:FP}(::ProbLink,x::T) = sqrt2*erfinv(convert(T,2.0)*x-one(T))
type ProbInv <: UnaryFunctor end
evaluate{T<:FP}(::ProbInv,x::T) = (one(T) + erf(x/sqrt2))/convert(T,2.0)
type ProbME <: UnaryFunctor end
evaluate{T<:FP}(::ProbME,x::T) = exp(-x*x/convert(T,2.0))/sqrt2pi

function linkfun!{T<:FP}(::Type{CauchitLink}, eta::Vector{T}, mu::Vector{T})
     map!(CauchLink(),eta,mu)
end
function linkinv!{T<:FP}(::Type{CauchitLink}, mu::Vector{T}, eta::Vector{T})
    map!(CauchInv(),mu,eta)
end
function mueta!{T<:FP}(::Type{CauchitLink}, me::Vector{T}, eta::Vector{T})
    map!(CauchME(),me,eta)
end

function linkfun!{T<:FP}(::Type{CloglogLink}, eta::Vector{T}, mu::Vector{T})
    map!(CLgLgLink(),eta,mu)
end
function linkinv!{T<:FP}(::Type{CloglogLink}, mu::Vector{T}, eta::Vector{T})
    map!(CLgLgInv(),mu,eta)
end
function mueta!{T<:FP}(::Type{CloglogLink}, me::Vector{T}, eta::Vector{T})
    map!(CLgLgME(),me,eta)
end

function linkfun!{T<:FP}(::Type{IdentityLink}, eta::Vector{T}, mu::Vector{T})
    copy!(eta,mu)
end
function linkinv!{T<:FP}(::Type{IdentityLink}, mu::Vector{T}, eta::Vector{T})
    copy!(mu,eta)
end
function mueta!{T<:FP}(::Type{IdentityLink}, me::Vector{T}, eta::Vector{T})
    fill!(me,one(T))
end

function linkfun!{T<:FP}(::Type{InverseLink}, eta::Vector{T}, mu::Vector{T})
    map!(Divide(),eta,one(T),mu)
end
function linkinv!{T<:FP}(::Type{InverseLink}, mu::Vector{T}, eta::Vector{T})
    map!(Divide(),mu,one(T),eta)
end
function mueta!{T<:FP}(::Type{InverseLink}, me::Vector{T}, eta::Vector{T})
    map!(InvME(),me,eta)
end

function linkfun!{T<:FP}(::Type{LogitLink}, eta::Vector{T}, mu::Vector{T})
    map!(Logit(),eta,mu)
end
function linkinv!{T<:FP}(::Type{LogitLink}, mu::Vector{T}, eta::Vector{T})
    map!(NumericExtensions.Logistic(),mu,eta)
end
function mueta!{T<:FP}(::Type{LogitLink}, me::Vector{T}, eta::Vector{T})
    map!(LogitME(),me,eta)
end

function linkfun!{T<:FP}(::Type{LogLink}, eta::Vector{T}, mu::Vector{T})
    map!(Log(),eta,mu)
end
function linkinv!{T<:FP}(::Type{LogLink}, mu::Vector{T}, eta::Vector{T})
    map!(Exp(),mu,eta)
end
function mueta!{T<:FP}(::Type{LogLink}, mu::Vector{T}, eta::Vector{T})
    map!(Exp(),mu,eta)
end

function linkfun!{T<:FP}(::Type{ProbitLink}, eta::Vector{T}, mu::Vector{T})
    map!(ProbLink(),eta,mu)
end
function linkinv!{T<:FP}(::Type{ProbitLink}, mu::Vector{T}, eta::Vector{T})
    map!(ProbInv(),mu,eta)
end
function mueta!{T<:FP}(::Type{ProbitLink}, me::Vector{T}, eta::Vector{T})
    map!(ProbME(),me,eta)
end

canonicallink(::Gamma) = InverseLink
canonicallink(::Normal) = IdentityLink
canonicallink(::Bernoulli) = LogitLink
canonicallink(::Poisson) = LogLink
canonicallink(::Type{Gamma}) = InverseLink
canonicallink(::Type{Normal}) = IdentityLink
canonicallink(::Type{Bernoulli}) = LogitLink
canonicallink(::Type{Poisson}) = LogLink

type BernoulliVar <: UnaryFunctor end
evaluate{T<:FP}(::BernoulliVar,x::T) = x * (one(T) - x)

var!{T<:FP}(::Type{Bernoulli},v::Vector{T},mu::Vector{T}) = map!(BernoulliVar(),v,mu)
var!{T<:FP}(::Type{Gamma},v::Vector{T},mu::Vector{T}) = map!(Abs2(),v,mu)
var!{T<:FP}(::Type{Normal},v::Vector{T},mu::Vector{T}) = fill!(v,one(T))
var!{T<:FP}(::Type{Poisson},v::Vector{T},mu::Vector{T}) = copy!(v,mu)

type BernoulliMuStart <: BinaryFunctor end
evaluate{T<:FP}(::BernoulliMuStart,y::T,w::T) = (w*y + convert(T,0.5))/(w + one(T))
result_type{T<:FP}(::BernoulliMuStart,::Type{T},::Type{T}) = T

mustart{T<:FP}(::Type{Bernoulli},y::Vector{T},wt::Vector{T}) = map(BernoulliMuStart(),y,wt)
mustart{T<:FP}(::Type{Gamma},y::Vector{T},wt::Vector{T}) = copy(y)
mustart{T<:FP}(::Type{Normal},y::Vector{T},wt::Vector{T}) = copy(y)
mustart{T<:FP}(::Type{Poisson},y::Vector{T},wt::Vector{T}) = map(Add(), y, convert(T,0.1))

for Op in [:BernoulliVar,
           :CauchLink, :CauchInv, :CauchME,
           :CLgLgLink, :CLgLgInv, :CLgLgME,
           :InvME,
           :LogitME,
           :ProbLink, :ProbInv, :ProbME]
    @eval result_type{T<:FP}(::$(Op), ::Type{T}) = T
end


type BernoulliDevResid <: TernaryFunctor end
function evaluate{T<:FP}(::BernoulliDevResid,y::T,mu::T,wt::T)
    omy = one(T)-y
    2.wt*(xlogy(y,y/mu) + xlogy(omy,omy/(one(T)-mu)))
end
result_type{T<:FP}(::BernoulliDevResid,::Type{T},::Type{T},::Type{T}) = T

type PoissonDevResid <: TernaryFunctor end
evaluate{T<:FP}(::PoissonDevResid,y::T,mu::T,wt::T) = 2.wt * (xlogy(y,y/mu) - (y - mu))
result_type{T<:FP}(::PoissonDevResid,::Type{T},::Type{T},::Type{T}) = T

function devresid!{T<:FP}(::Type{Bernoulli},dr::Vector{T},y::Vector{T},
                                     mu::Vector{T},wt::Vector{T})
    map!(BernoulliDevResid(), dr, y, mu, wt)
end
function devresid!{T<:FP}(::Type{Poisson},dr::Vector{T},y::Vector{T},
                                     mu::Vector{T},wt::Vector{T})
    map!(PoissonDevResid(), dr, y, mu, wt)
end
