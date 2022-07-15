## To remove when https://github.com/JuliaStats/StatsAPI.jl/pull/16 is merged

mdisp(x::LmResp) = one()
mdisp(rr::GlmResp{T1, <: Normal, T2, T3}) where {T1, T2, T3} = one()

function mdisp(rr::GlmResp{T1, <: Union{Gamma, Bernoulli, InverseGaussian}, T2, T3}) where {T1, T2, T3}
    sum(abs2, rr.wrkwt.*rr.wrkresid)/sum(rr.wrkwt)
end


function momentmatrix(model::RegressionModel; weighted::Bool=false) 
    X = modelmatrix(model; weighted=weightd)
    d = mdisp(model.model.rr)
    r = residuals(model; weighted=weightd)
    return (X.*r)./d
end

# function momentmatrix(model::RegressionModel; weighted::Bool=false) 
#     X = modelmatrix(model; weighted=false)
#     mm = similar(X)
#     if weighted
#         r = residual()
#     d = dispersion(model)
#     r = residuals(model; weighted=false)
#     return (X.*r)./d
# end

