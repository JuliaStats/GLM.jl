# Generalized linear models (glm's) in Julia

To use the package, clone this repository and, in Julia, run
```julia
push(LOAD_PATH, "/path/to/repository/src/")
require("init.jl")
```

The `glmFit` function in this package fits generalized linear models with the Iteratively Reweighted Least Squares (IRLS) algorithm.  It is closer to the R function glm.fit than to R's glm in that the user is required to specify the model matrix as a matrixm, rather than in a formula/data specification.

A `GlmResp` object is created from the response vector, distribution and, optionally, the link.  The available distributions and their canonical link functions are

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

    X = float64([1 0 0 0 0; 1 1 0 0 0; 1 0 1 0 0; 1 0 0 1 0; 1 1 0 1 0; 1 0 1 1 0; 1 0 0 0 1; 1 1 0 0 1; 1 0 1 0 1])

and the fit is

    d = DensePred(X)
    glmFit(d, rr)
    d.beta
    deviance(rr)


