# Implementation

## Separation of response object and predictor object

The general approach in this code is to separate functionality related
to the response from that related to the linear predictor. This
allows for greater generality by mixing and matching different
subtypes of the abstract type `LinPred` and the abstract type `ModResp`.

A `LinPred` type incorporates the parameter vector and the model
matrix. The parameter vector is a dense numeric vector but the model
matrix can be dense or sparse. A `LinPred` type must incorporate
some form of a decomposition of the weighted model matrix that allows
for the solution of a system `X'W * X * delta=X'wres` where `W` is a
diagonal matrix of "X weights", provided as a vector of the square
roots of the diagonal elements, and `wres` is a weighted residual vector.

Currently there are two dense predictor types, `DensePredQR` and
`DensePredChol`, and the usual caveats apply. The Cholesky
version is faster but somewhat less accurate than that QR version.
The skeleton of a distributed predictor type is in the code
but not yet fully fleshed out. Because Julia by default uses
OpenBLAS, which is already multi-threaded on multicore machines, there
may not be much advantage in using distributed predictor types.

A `ModResp` type must provide methods for the `wtres` and
`sqrtxwts` generics. Their values are the arguments to the
`updatebeta` methods of the `LinPred` types. The
`Float64` value returned by `updatedelta` is the value of the
convergence criterion.

Similarly, `LinPred` types must provide a method for the
`linpred` generic. In general `linpred` takes an instance of
a `LinPred` type and a step factor. Methods that take only an instance
of a `LinPred` type use a default step factor of 1. The value of
`linpred` is the argument to the `updatemu` method for
`ModResp` types. The `updatemu` method returns the updated
deviance.