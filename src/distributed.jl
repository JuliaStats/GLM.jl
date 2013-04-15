#export DistPred

## type DGlmResp                    # distributed response in a glm model
##     d::Distribution
##     l::Link
##     eta::DArray{Float64,1,1}     # linear predictor
##     mu::DArray{Float64,1,1}      # mean response
##     offset::DArray{Float64,1,1}  # offset added to linear predictor (usually 0)
##     wts::DArray{Float64,1,1}     # prior weights
##     y::DArray{Float64,1,1}       # response
##     ## FIXME: Add compatibility checks here
## end

## function DGlmResp(d::Distribution, l::Link, y::DArray{Float64,1,1})
##     wt     = darray((T,d,da)->ones(T,d), Float64, size(y), distdim(y), y.pmap)
##     offset = darray((T,d,da)->zeros(T,d), Float64, size(y), distdim(y), y.pmap)
##     mu     = similar(y)
##     @sync begin
##         for p = y.pmap
##             @spawnat p copy_to(localize(mu), d.mustart(localize(y), localize(wt)))
##         end
##     end
##     dGlmResp(d, l, map_vectorized(link.linkFun, mu), mu, offset, wt, y)
## end

## DGlmResp(d::Distribution, y::DArray{Float64,1,1}) = DGlmResp(d, canonicallink(d), y)

## At_mul_B{T <: Real}(A::DArray{T, 2, 1}, B::DArray{T, 2, 1}) = Ac_mul_B(A, B)

## function Ac_mul_B{T <: Real}(A::DArray{T, 2, 1}, B::DArray{T, 2, 1})
##     if (all(procs(A) != procs(B)) || all(dist(A) != dist(B)))
##         # FIXME: B should be redistributed to match A
##         error("Arrays A and B must be distributed similarly")
##     end
##     if is(A, B)
##         return mapreduce(+, fetch, {@spawnat p BLAS.syrk('T', localize(A)) for p in procs(A)})
##     end
##     mapreduce(+, fetch, {@spawnat p Ac_mul_B(localize(A), localize(B)) for p in procs(A)})
## end

## function Ac_mul_B{T <: Real}(A::DArray{T, 2, 1}, B::DArray{T, 1, 1})
##     if (all(procs(A) != procs(B)) || all(dist(A) != dist(B)))
##         # FIXME: B should be redistributed to match A
##         error("Arrays A and B must be distributed similarly")
##     end
##     mapreduce(+, fetch, {@spawnat p Ac_mul_B(localize(A), localize(B)) for p in procs(A)})
## end

## type DistPred{T} <: LinPred   # predictor with distributed (on rows) X
##     X::DArray{T, 2, 1}        # model matrix
##     beta::Vector{T}           # coefficient vector
##     r::CholeskyDense{T}
##     function DistPred(X, beta)
##         if size(X, 2) != length(beta) error("dimension mismatch") end
##         new(X, beta, chold(X'X))
##     end
## end

## function (\)(A::DArray{Float64,2,1}, B::DArray{Float64,1,1})
##     R   = Cholesky(A'A)
##     LAPACK.potrs!('U', R, A'B)
## end
