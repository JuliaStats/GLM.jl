function glmFit(p::DensePred, r::GlmResp, maxIter::Uint, minStepFac::Float64, convTol::Float64)
    if (maxIter < 1) error("maxIter must be positive") end
    if !(0 < minStepFac < 1) error("minStepFac must be in (0, 1)") end
    cvg = false

    devold = typemax(Float64)           # Float64 version of Inf
    for i=1:maxIter
        dev = updateMu(r, linPred(updateBeta(p, wrkResp(r), sqrtWrkWt(r))))
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

