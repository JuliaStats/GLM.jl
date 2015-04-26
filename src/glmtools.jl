using Base.Cartesian

# We use simdmap! instead of map! because we need to support 0.3, where
# we have to call evaluate(f, x) instead of f(x). Also, even on 0.4,
# map! is slow when given >2 input arrays. Finally, we can use @simd
# here, although it's unclear how big the performance advantage really
# is. Revisit when we drop support for julia 0.3 and
# https://github.com/JuliaLang/julia/issues/9900 is fixed.
@ngenerate N typeof(A) function simdmap!(f, A::AbstractArray, Xs::NTuple{N,AbstractArray}...)
    @nexprs N d->(length(Xs_d) == length(A) || throw(DimensionMismatch()))
    @inbounds @simd for i = 1:length(A)
        @nexprs N k->(v_k = Xs_k[i])
        A[i] = @ncall N evaluate f v
    end
    A
end
# stagedfunction simdmap!(f, A::AbstractArray, Xs::AbstractArray...)
#     lengthcheck = Expr(:comparison, :(length(A)))
#     for i = 1:length(Xs)
#         push!(lengthcheck.args, :(==))
#         push!(lengthcheck.args, :(length(Xs[$i])))
#     end
#     quote
#         $lengthcheck || throw(DimensionMismatch())
#         @inbounds @simd for i = 1:length(A)
#             A[i] = $(Expr(:call, :evaluate, :f, [:(Xs[$j][i]) for j = 1:length(Xs)]...))
#         end
#         A
#     end
# end

abstract Link             # Link types define linkfun!, linkinv!, and mueta!

type CauchitLink <: Link end
type CloglogLink  <: Link end
type IdentityLink <: Link end
type InverseLink  <: Link end
type LogitLink <: Link end
type LogLink <: Link end
type ProbitLink <: Link end
type SqrtLink <: Link end

type CauchLink <: Functor{1} end
evaluate{T<:FP}(::CauchLink,x::T) = tan(pi*(x - convert(T,0.5)))
type CauchInv <: Functor{1} end
evaluate{T<:FP}(::CauchInv,x::T) = convert(T,0.5) + atan(x)/pi
type CauchME <: Functor{1} end
evaluate{T<:FP}(::CauchME,x::T) = one(T)/(pi*(one(T) + abs2(x)))
type CLgLgLink <: Functor{1} end
evaluate{T<:FP}(::CLgLgLink,x::T) = log(-log1p(-x))
type CLgLgInv <: Functor{1} end
evaluate{T<:FP}(::CLgLgInv,x::T) = -expm1(-exp(x))
type CLgLgME <: Functor{1} end
evaluate{T<:FP}(::CLgLgME,x::T) = exp(x)*exp(-exp(x))
type InvME <: Functor{1} end
evaluate{T<:FP}(::InvME,x::T) = -one(T)/abs2(x)
type LogitME <: Functor{1} end
evaluate{T<:FP}(::LogitME,x::T) = (e = exp(-abs(x)); f = one(T) + e; e / (f * f))
type ProbLink <: Functor{1} end
evaluate{T<:FP}(::ProbLink,x::T) = -sqrt2*erfcinv(2*x)
type ProbInv <: Functor{1} end
evaluate{T<:FP}(::ProbInv,x::T) = erfc(-x/sqrt2)/2
type ProbME <: Functor{1} end
evaluate{T<:FP}(::ProbME,x::T) = exp(-x*x*convert(T,0.5))/sqrt2π
type SqrtME <: Functor{1} end
evaluate{T<:FP}(::SqrtME,x::T) = 2*x

type LogisticFun <: Functor{1} end
evaluate{T<:FP}(::LogisticFun,x::T) = one(x) / (one(x) + exp(-x))


# IdentityLink is a special case - nothing to map
linkfun!{T<:FP}(::IdentityLink, eta::Vector{T}, mu::Vector{T}) = copy!(eta,mu)
linkinv!{T<:FP}(::IdentityLink, mu::Vector{T}, eta::Vector{T}) = copy!(mu,eta)
mueta!{T<:FP}(::IdentityLink, me::Vector{T}, eta::Vector{T}) = fill!(me,one(T))

for (l, lf, li, mueta) in
    ((:CauchitLink, :CauchLink, :CauchInv, :CauchME),
     (:CloglogLink, :CLgLgLink, :CLgLgInv, :CLgLgME),
     (:InverseLink, :RcpFun, :RcpFun, :InvME),
     (:LogitLink, :LogitFun, :LogisticFun, :LogitME),
     (:LogLink, :LogFun, :ExpFun, :ExpFun),
     (:ProbitLink, :ProbLink, :ProbInv, :ProbME),
     (:SqrtLink, :SqrtFun, :Abs2Fun, :SqrtME))
    @eval begin
        linkfun!{T<:FP}(::$l,eta::Vector{T},mu::Vector{T}) = simdmap!($lf(),eta,mu)
        linkinv!{T<:FP}(::$l,mu::Vector{T},eta::Vector{T}) = simdmap!($li(),mu,eta)
        mueta!{T<:FP}(::$l,me::Vector{T},eta::Vector{T}) = simdmap!($mueta(),me,eta)
    end
end

canonicallink(::Gamma) = InverseLink()
canonicallink(::Normal) = IdentityLink()
canonicallink(::Binomial) = LogitLink()
canonicallink(::Poisson) = LogLink()

# For the "odd" link functions we evaluate the linear predictor such that mu is closest to zero where the precision is higher
for (l, li) in
    ((:CauchitLink, :CauchInv),
     (:InverseLink, :RcpFun),
     (:LogitLink, :LogisticFun),
     (:ProbitLink, :ProbInv))
    @eval begin
        function var!{T<:FP}(::Binomial,l::$l,v::Vector{T},eta::Vector{T})
            @inbounds @simd for i = 1:length(eta)
                η = eta[i]
                mu = evaluate($li(), ifelse(η < 0, η, -η))
                v[i] = mu*(1 - mu)
            end
            v
        end
    end
end
# but Cloglog is not "odd"
function var!{T<:FP}(::Binomial,l::CloglogLink,v::Vector{T},eta::Vector{T})
    @inbounds @simd for i = 1:length(eta)
        mu = evaluate(CLgLgInv(), eta[i])
        v[i] = mu*(1 - mu)
    end
    v
end

var!{T<:FP}(::Gamma,v::Vector{T},mu::Vector{T}) = simdmap!(Abs2Fun(),v,mu)
var!{T<:FP}(::Normal,v::Vector{T},mu::Vector{T}) = fill!(v,one(T))
var!{T<:FP}(::Poisson,v::Vector{T},mu::Vector{T}) = copy!(v,mu)

type BinomialMuStart <: Functor{2} end
evaluate{T<:FP}(::BinomialMuStart,y::T,w::T) = (w*y + convert(T,0.5))/(w + one(T))

mustart!{T<:FP}(::Binomial,mu::Vector{T},y::Vector{T},wt::Vector{T}) = simdmap!(BinomialMuStart(),mu,y,wt)
mustart!{T<:FP}(::Gamma,mu::Vector{T},y::Vector{T},::Vector{T}) = copy!(mu, y)
mustart!{T<:FP}(::Normal,mu::Vector{T},y::Vector{T},::Vector{T}) = copy!(mu, y)
mustart!{T<:FP}(::Poisson,mu::Vector{T},y::Vector{T},::Vector{T}) = broadcast!(+, mu, y, convert(T,0.1))
mustart{T<:FP}(d::Distribution,y::Vector{T},wt::Vector{T}) = mustart!(d, similar(y), y, wt)

type BinomialDevResid <: Functor{3} end
function evaluate{T<:FP}(::BinomialDevResid,y::T,mu::T,wt::T)
    if y == 1
        return 2.0*wt*-log(mu)
    elseif y == 0
        return -2.0*wt*log1p(-mu)
    else
        return 2.0*wt*(y*(log(y) - log(mu)) + (1 - y)*(log1p(-y) - log1p(-mu)))
    end
end

type PoissonDevResid <: Functor{3} end
evaluate{T<:FP}(::PoissonDevResid,y::T,mu::T,wt::T) = 2.wt * (StatsBase.xlogy(y,y/mu) - (y - mu))

type GammaDevResid <: Functor{3} end
evaluate{T<:FP}(::GammaDevResid,y::T,mu::T,wt::T) = -2.wt * (log(y/mu) - (y - mu)/mu)

type GaussianDevResid <: Functor{3} end
evaluate{T<:FP}(::GaussianDevResid,y::T,mu::T,wt::T) = wt * abs2(y - mu)

function devresid!{T<:FP}(::Binomial,dr::Vector{T},y::Vector{T},
                                     mu::Vector{T},wt::Vector{T})
    simdmap!(BinomialDevResid(), dr, y, mu, wt)
end
function devresid!{T<:FP}(::Poisson,dr::Vector{T},y::Vector{T},
                                     mu::Vector{T},wt::Vector{T})
    simdmap!(PoissonDevResid(), dr, y, mu, wt)
end
function devresid!{T<:FP}(::Gamma,dr::Vector{T},y::Vector{T},
                                  mu::Vector{T},wt::Vector{T})
    simdmap!(GammaDevResid(), dr, y, mu, wt)
end
function devresid!{T<:FP}(::Normal,dr::Vector{T},y::Vector{T},
                                     mu::Vector{T},wt::Vector{T})
    simdmap!(GaussianDevResid(), dr, y, mu, wt)
end
