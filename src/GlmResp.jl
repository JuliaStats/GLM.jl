## GlmResp and dGlmResp types.
## FIXME: Add an abstract type with these as concrete subtypes?

const VF64 = Vector{Float64}

type GlmResp                            # response in a glm model
    d::Distribution                  
    l::Link
    eta::VF64              # linear predictor
    mu::VF64               # mean response
    offset::VF64           # offset added to linear predictor (usually 0)
    wts::VF64              # prior weights
    y::VF64                # response
    function GlmResp(dd,ll,ee::VF64,mm::VF64,oo::VF64,ww::VF64,yy::VF64)
        if !(numel(ee) == numel(mm) == numel(oo) == numel(ww) == numel(yy))
            error("mismatched sizes")
        end
        insupport(dd, yy)? new(dd,ll,ee,mm,oo,ww,yy): error("elements of y not in distribution support")
    end
end

## outer constructor - the most common way of creating the object
function GlmResp(d::Distribution, l::Link, y::VF64)
    sz = size(y)
    wt = ones(Float64, sz)
    mu = mustart(d, y, wt)
    GlmResp(d, l, linkfun(l, mu), mu, zeros(Float64, sz), wt, y)
end

## another outer constructor using the canonical link for the distribution
GlmResp(d::Distribution, y::VF64) = GlmResp(d, canonicalLink(d), y)

deviance( r::GlmResp) = deviance(r.d, r.mu, r.y, r.wts)
devResid( r::GlmResp) = devResid(r.d, r.y, r.mu, r.wts)
drsum(    r::GlmResp) = sum(devResid(r))
mueta(    r::GlmResp) = mueta(r.l, r.eta)
sqrtWrkWt(r::GlmResp) = mueta(r) .* sqrt(r.wts ./ var(r))
var(      r::GlmResp) = var(r.d, r.mu)
wrkResid( r::GlmResp) = (r.y - r.mu) ./ mueta(r)
wrkResp(  r::GlmResp) = (r.eta - r.offset) + wrkResid(r)

function updateMu{T<:Real}(r::GlmResp, linPr::AbstractArray{T})
    promote_shape(size(linPr), size(r.eta)) # size check
    for i=1:numel(linPr)
        r.eta[i] = linPr[i] + r.offset[i]
        r.mu[i]  = linkinv(r.l, r.eta[i])
    end
    deviance(r)
end
    
type dGlmResp                    # distributed response in a glm model
    dist::Distribution
    link::Link
    eta::DArray{Float64,1,1}     # linear predictor
    mu::DArray{Float64,1,1}      # mean response
    offset::DArray{Float64,1,1}  # offset added to linear predictor (usually 0)
    wts::DArray{Float64,1,1}     # prior weights
    y::DArray{Float64,1,1}       # response
    ## FIXME: Add compatibility checks here
end

## outer constructor - the most common way of creating the object
function dGlmResp(d::Distribution, link::Link, y::DArray{Float64,1,1})
    wt     = darray((T,d,da)->ones(T,d), Float64, size(y), distdim(y), y.pmap)
    offset = darray((T,d,da)->zeros(T,d), Float64, size(y), distdim(y), y.pmap)
    mu     = similar(y)
    @sync begin
        for p = y.pmap
            @spawnat p copy_to(localize(mu), d.mustart(localize(y), localize(wt)))
        end
    end
    dGlmResp(d, link, map_vectorized(link.linkFun, mu), mu, offset, wt, y)
end

## another outer constructor using the canonical link for the distribution
dGlmResp(d::Distribution, y::DArray{Float64,1,1}) = dGlmResp(d, canonicalLink(d), y)

