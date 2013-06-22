using NumericFunctors
import NumericFunctors: evaluate, result_type
import Base: var

typealias EwiseArray Union(Array,BitArray)

type Logit <: UnaryFunctor end
evaluate(::Logit, mu) = log(mu/(1.-mu))
result_type(::Logit,t::Type) = NumericFunctors.to_fptype(t)

type Logistic <: UnaryFunctor end
evaluate(::Logistic, eta) = 1./(1.+exp(-eta))
result_type(::Logistic,t::Type) = NumericFunctors.to_fptype(t)

type BernoulliVar <: UnaryFunctor end
evaluate(::BernoulliVar, mu) = mu*(1.-mu)
result_type(::BernoulliVar,t::Type) = NumericFunctors.to_fptype(t)

type BernoulliWeights <: UnaryFunctor end
evaluate(::BernoulliWeights, mu) = 1./(mu*(1.-mu))
result_type(::BernoulliWeights,t::Type) = NumericFunctors.to_fptype(t)

type WeightedSqrDiff <: TernaryFunctor end
evaluate(::WeightedSqrDiff, x, y, w) = w * abs2(x - y)
result_type(::WeightedSqrDiff,t1::Type,t2::Type,t3::Type) = NumericFunctors.to_fptype(promote_type(t1,t2,t3))

type LogisticRegression{T<:Float64}
    X::Matrix{T}
    y::BitArray{1}
    beta::Vector{T}
    mu::Vector{T}
    eta::Vector{T}
    weights::Vector{T}
end

function LogisticRegression(X::Matrix,y)
    Xm = float(X); yy = convert(BitArray{1},y)
    n,p = size(Xm); length(yy) == n || error("Dimension mismatch")
    mu = Float64[v ? 0.75 : 0.25 for v in yy]
    LogisticRegression(Xm, yy, zeros(p), mu, map(Logit(),mu), ones(n))
end

link(l::LogisticRegression) = map!(Logit(), l.eta, l.mu)
linkinv(l::LogisticRegression) = map!(Logistic(), l.mu, l.eta)
weights(l::LogisticRegression) = map!(BernoulliWeights(), l.weights, l.mu)
wsqrdiff(x::EwiseArray, y::EwiseArray, w::EwiseArray) = mapreduce(WeightedSqrDiff(), Add(), x, y, w)
