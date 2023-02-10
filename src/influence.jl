#####
##### Measures of influence
#####

StatsBase.leverage(model::LinPredModel) = _leverage(model, model.pp)

function _leverage(_, pred::DensePredQR)
    Q = pred.qr.Q
    r = linpred_rank(pred)
    y = diagm(size(Q, 1), r, trues(r))
    Z = Q * y
    return vec(sum(abs2, Z; dims=1))
end

function _leverage(model, pred::DensePredChol{<:Any,<:Cholesky})
    X = weightedmodelmatrix(model)
    choldiv!(pred.chol, X)
    return vec(sum(abs2, X; dims=2))
end

function _leverage(model, pred::DensePredChol{<:Any,<:CholeskyPivoted})
    X = weightedmodelmatrix(model)
    C = pred.chol
    if any(x -> isapprox(x, zero(x)), diag(C.L))
        Q = qr!(X).Q
        r = rank(C)
        y = diagm(size(Q, 1), r, trues(r))
        X = Q * y
    else
        choldiv!(C, X)
    end
    return vec(sum(abs2, X; dims=2))
end

function _leverage(model, pred::SparsePredChol)
    X = weightedmodelmatrix(model)
    Z = pred.chol.L \ X'  # can't be done in-place for SuiteSparse factorizations
    return vec(sum(abs2, Z; dims=1))
end

# Overwrite `X` with the solution `Z` to `L*Z = Xᵀ`, where `L` is the lower triangular
# factor from the Cholesky factorization of `XᵀX`.
choldiv!(C::Cholesky, X) = ldiv!(C.L, X')
choldiv!(C::CholeskyPivoted, X) = ldiv!(C.L, view(X, :, invperm(C.p))')

# `X` for unweighted models, `√W*X` for weighted, where `W` is a diagonal matrix
# of the prior weights. A copy of `X` is made so that it can be mutated downstream
# without affecting the underlying model object.
function weightedmodelmatrix(model::LinearModel)
    X = copy(modelmatrix(model))
    priorwt = model.rr.wts
    if !isempty(priorwt)
        X .*= sqrt.(priorwt)
    end
    return X
end

# `√W*X`, where `W` is a diagonal matrix of the working weights from the final IRLS
# iteration. This handles GLMs with and without prior weights, as the prior weights
# simply become part of the working weights for IRLS. No explicit copy of `X` needs
# to be made since we're always doing a multiplication, unlike for the method above.
weightedmodelmatrix(model::GeneralizedLinearModel) =
    modelmatrix(model) .* sqrt.(model.rr.wrkwt)

@noinline function _checkrankdeficient(model)
    if linpred_rank(model.pp) < size(modelmatrix(model), 2)
        throw(ArgumentError("models with collinear terms are not currently supported"))
    end
    return nothing
end

function StatsBase.cooksdistance(model::GeneralizedLinearModel)
    _checkrankdeficient(model)
    y = response(model)
    ŷ = fitted(model)
    k = dof(model) - hasintercept(model)
    φ̂ = dispersion(model)
    h = leverage(model)
    D = model.rr.d
    return @. (y - ŷ)^2 / glmvar(D, ŷ) * (h / (1 - h)^2) / (φ̂ * (k + 1))
end

function StatsBase.cooksdistance(model::LinearModel)
    _checkrankdeficient(model)
    u = residuals(model)
    #if !isempty(model.rr.wts)
    #    u .*= sqrt.(model.rr.wts)
    #end
    mse = dispersion(model, true)
    k = dof(model) - 1
    h = leverage(model)
    return @. u^2 * (h / (1 - h)^2) / (k * mse)
end
