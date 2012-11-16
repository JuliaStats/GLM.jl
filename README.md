# Generalized linear models (glm's) in Julia

To use the package, clone this repository and, in Julia, run
```julia
require("/path/to/repository/src/Glm.jl")
using Distributions
using Glm
```

This will soon change when ```Glm.jl``` becomes a Julia package.

The `glmfit` function in this package fits generalized linear models
with the Iteratively Reweighted Least Squares (IRLS) algorithm.  It is
closer to the R function ```glm.fit``` than to R's ```glm``` in that
the user is required to specify the model matrix explicitly, rather
than implicitly in a formula/data specification.

A `GlmResp` object is created from the response vector, distribution
and, optionally, the link.  The available distributions and their
canonical link functions are

    Bernoulli (LogitLink)
    Poisson (LogLink)

and the available links are

    CauchitLink
    CloglogLink
    IdentityLink
    InverseLink
    LogitLink
    LogLink
    ProbitLink

The response in the example on p. 93 of Dobson (1990) would be written

    rr = GlmResp(Poisson(), [18.,17,15,20,10,20,25,13,12])

At present the model matrix must be generated in the following awkward way

	ct3 = contr_treatment(3)
	pp = DensePredQR(hcat(ones(Float64,9), indicators(gl(3,1,9))[1]*ct3, indicators(gl(3,3))[1]*ct3))

and the fit is

	julia> glmfit(pp, rr)  # output is iteration: deviance, convergence criterion
	1: 46.81189638788046, Inf
	2: 46.76132443472076, 0.0010814910349747592
	3: 46.761318401957794, 1.2901182375636843e-7

	julia> pp.beta
	5-element Float64 Array:
	 3.04452   
	-0.454255  
    -0.292987  
     1.92349e-8
     8.38339e-9

	julia> deviance(rr)
	46.761318401957794

There are two dense predictor representations, ```DensePredQR``` and
```DensePredChol```, and the usual caveats apply.  The Cholesky
version is faster but somewhat less accurate than that QR version.
The skeleton of a distributed predictor representation is in the code
but not yet fully fleshed out.

Other examples are shown in ```test/glmFit.jl```.
