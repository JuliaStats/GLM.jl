function glmFit(p::DensePred, r::GlmResp, maxIter::Uint, minStepFac::Float64, convTol::Float64)
    if (maxIter < 1) error("maxIter must be positive") end
    if !(0 < minStepFac < 1) error("minStepFac must be in (0, 1)") end
    cvg = false

    devold = typemax(Float64)           # Float64 version of Inf
    for i=1:maxIter
        updateBeta(p, wrkResp(r), sqrtWrkWt(r))
        dev = updateMu(r, linPred(p))
        println("old: $devold, dev = $dev")
        if (dev >= devold)
            error("code needed to handle the step-factor case")
        end
        if abs((devold - dev)/dev) < convTol
            cvg = true
            break
        end
        devold = dev
    end
    if !cvg
        error("failure to converge in $maxIter iterations")
    end
end

glmFit(p::DensePred, r::GlmResp) = glmFit(p, r, uint(30), 0.001, 1.e-6)

function glm(f::Formula, df::DataFrame, d::Distribution, l::Link)
    mm = model_matrix(f, df)
    rr = GlmResp(d, l, vec(mm.response))
    dp = DensePred(mm.model)
    glmFit(dp, rr)
end

glm(f::Formula, df::DataFrame, d::Distribution) = glm(f, df, d, canonicalLink(d))

glm(f::Expr, df::DataFrame, d::Distribution) = glm(Formula(f), df, d)
