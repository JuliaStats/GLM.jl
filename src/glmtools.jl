abstract Link             # Link types define linkfun!, linkinv!, and mueta!

import Base.@math_const

@math_const sqrt2   1.4142135623730950488 sqrt(big(2.))
@math_const sqrt2pi 2.5066282746310005024 sqrt(big(2.)*π)

type CauchitLink <: Link end
type CloglogLink  <: Link end
type IdentityLink <: Link end
type InverseLink  <: Link end
type LogitLink <: Link end
type LogLink <: Link end
type ProbitLink <: Link end

type CauchLink <: Functor{1} end
evaluate{T<:FP}(::CauchLink,x::T) = tan(pi*(x - convert(T,0.5)))
type CauchInv <: Functor{1} end
evaluate{T<:FP}(::CauchInv,x::T) = convert(T,0.5) + atan(x)/pi
type CauchME <: Functor{1} end
evaluate{T<:FP}(::CauchME,x::T) = one(T)/(pi*(one(T) + abs2(x)))
type CLgLgLink <: Functor{1} end
evaluate{T<:FP}(::CLgLgLink,x::T) = log(-log(one(T) - x))
type CLgLgInv <: Functor{1} end
evaluate{T<:FP}(::CLgLgInv,x::T) = -expm1(-exp(x))
type CLgLgME <: Functor{1} end
evaluate{T<:FP}(::CLgLgME,x::T) = exp(x)*exp(-exp(x))
type InvME <: Functor{1} end
evaluate{T<:FP}(::InvME,x::T) = -one(T)/abs2(x)
type LogitME <: Functor{1} end
evaluate{T<:FP}(::LogitME,x::T) = (e = exp(-abs(x)); f = one(T) + e; e / (f * f))
type ProbLink <: Functor{1} end
evaluate{T<:FP}(::ProbLink,x::T) = sqrt2*erfinv(convert(T,2.0)*x-one(T))
type ProbInv <: Functor{1} end
evaluate{T<:FP}(::ProbInv,x::T) = (one(T) + erf(x/sqrt2))/convert(T,2.0)
type ProbME <: Functor{1} end
evaluate{T<:FP}(::ProbME,x::T) = exp(-x*x/convert(T,2.0))/sqrt2pi

                                        # IdentityLink is a special case - nothing to map
linkfun!{T<:FP}(::IdentityLink, eta::Vector{T}, mu::Vector{T}) = copy!(eta,mu)
linkinv!{T<:FP}(::IdentityLink, mu::Vector{T}, eta::Vector{T}) = copy!(mu,eta)
mueta!{T<:FP}(::IdentityLink, me::Vector{T}, eta::Vector{T}) = fill!(me,one(T))

for (l, lf, li, mueta) in
    ((:CauchitLink, :CauchLink, :CauchInv, :CauchME),
     (:CloglogLink, :CLgLgLink, :CLgLgInv, :CLgLgME),
     (:InverseLink, :Recip, :Recip, :InvME),
     (:LogitLink, :LogitFun, :LogisticFun, :LogitME),
     (:LogLink, :LogFun, :ExpFun, :ExpFun),
     (:ProbitLink, :ProbLink, :ProbInv, :ProbME))
    @eval begin
        linkfun!{T<:FP}(::$l,eta::Vector{T},mu::Vector{T}) = map!($lf(),eta,mu)
        linkinv!{T<:FP}(::$l,mu::Vector{T},eta::Vector{T}) = map!($li(),mu,eta)
        mueta!{T<:FP}(::$l,me::Vector{T},eta::Vector{T}) = map!($mueta(),me,eta)
    end
end

canonicallink(::Gamma) = InverseLink()
canonicallink(::Normal) = IdentityLink()
canonicallink(::Binomial) = LogitLink()
canonicallink(::Poisson) = LogLink()

type BernoulliVar <: Functor{1} end
evaluate{T<:FP}(::BernoulliVar,x::T) = x * (one(T) - x)

var!{T<:FP}(::Binomial,v::Vector{T},mu::Vector{T}) = map!(BernoulliVar(),v,mu)
var!{T<:FP}(::Gamma,v::Vector{T},mu::Vector{T}) = map!(Abs2Fun(),v,mu)
var!{T<:FP}(::Normal,v::Vector{T},mu::Vector{T}) = fill!(v,one(T))
var!{T<:FP}(::Poisson,v::Vector{T},mu::Vector{T}) = copy!(v,mu)

type BinomialMuStart <: Functor{2} end
evaluate{T<:FP}(::BinomialMuStart,y::T,w::T) = (w*y + convert(T,0.5))/(w + one(T))
result_type{T<:FP}(::BinomialMuStart,::Type{T},::Type{T}) = T

mustart{T<:FP}(::Binomial,y::Vector{T},wt::Vector{T}) = map(BinomialMuStart(),y,wt)
mustart{T<:FP}(::Gamma,y::Vector{T},::Vector{T}) = copy(y)
mustart{T<:FP}(::Normal,y::Vector{T},::Vector{T}) = copy(y)
mustart{T<:FP}(::Poisson,y::Vector{T},::Vector{T}) = y + convert(T,0.1)

for Op in [:BernoulliVar,
           :CauchLink, :CauchInv, :CauchME,
           :CLgLgLink, :CLgLgInv, :CLgLgME,
           :InvME,
           :LogitME,
           :ProbLink, :ProbInv, :ProbME]
    @eval result_type{T<:FP}(::$(Op), ::Type{T}) = T
end


type BinomialDevResid <: Functor{3} end
function evaluate{T<:FP}(::BinomialDevResid,y::T,mu::T,wt::T)
    omy = one(T)-y
    2.wt*(xlogy(y,y/mu) + xlogy(omy,omy/(one(T)-mu)))
end
result_type{T<:FP}(::BinomialDevResid,::Type{T},::Type{T},::Type{T}) = T

type PoissonDevResid <: Functor{3} end
evaluate{T<:FP}(::PoissonDevResid,y::T,mu::T,wt::T) = 2.wt * (xlogy(y,y/mu) - (y - mu))
result_type{T<:FP}(::PoissonDevResid,::Type{T},::Type{T},::Type{T}) = T

function devresid!{T<:FP}(::Binomial,dr::Vector{T},y::Vector{T},
                                     mu::Vector{T},wt::Vector{T})
    map!(BinomialDevResid(), dr, y, mu, wt)
end
function devresid!{T<:FP}(::Poisson,dr::Vector{T},y::Vector{T},
                                     mu::Vector{T},wt::Vector{T})
    map!(PoissonDevResid(), dr, y, mu, wt)
end
