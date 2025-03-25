# Supported Models

Broadly, there are two types of models provided by this package: linear models and generalized linear models (GLM).
In fact, (regular) linear models are just a special case of GLM.

## Linear Model

Suppose we have a response variable $y_i$ and explanatory variables $x_{1i}, x_{2i}, ...$
In its simplest form, the linear model uses the following:
```math
(Y_i\ \vert \ \mathbf{X}_i = \mathbf{x}_i)\ \sim\ \mathcal{N}(\mathbf{x}_i^\top \mathbb{\beta}, \sigma^2)
```


### Weighted Linear Model

This package also supports weighted ....

## Generalized Linear Model (GLM)

For GLMs, we have the same setup as before however the response $y_i$ is (usually) constrained in some way.
For instance, if we have count data $y_i$'s are non-negative integers.
GLMs model the parameters through a transformation of the mean that _links_ it to a linear predictor.
The function that operationalizes this is _link_ function (usually denoted by $g$).
```math
g(\mathbb{E}(Y_i | \mathbf{x}_i)) = \mathbf{x}_i^\top \mathbb{\beta} = \eta_i .
```
The link function must be invertible, and the inverse link function is also sometimes referred to as the mean function as it translates the linear predictor into the mean, that is:
```math
    \mathbb{E}(Y_i | \mathbf{x}_i) = g^{-1}(\eta_i) .
```

Thus, the link function allows us to go from the mean to the associated linear predictor and the mean function allows us to go from the linear predictor to the mean.


## Binary Response

A binary response $y_i$ can be modeled as:
```math
y_i \sim \operatorname{Bernoulli}(p_i) ,
```
where the $p_i$ is the mean, i.e. $\mathbb{E}(Y_i | \mathbf{x}_i) = \mathbb{P}(Y_i = 1 | \mathbf{x}_i) = p_i$ is modeled with a linear predictor using a link function as:
```math
g(p_i) =  \mathbf{x}_i^\top \mathbb{\beta}.
```

To fit a GLM for binary responses, use:
```julia
mod = glm(@formula(...), data, Bernoulli())
```

Note that the above function call did not specify a link function.
That is because, the _canonical_ link function is used as the default.
Briefly, the canonical link function simplifies the model.  

### Logistic Regression: The canonical link function

The _canonical_ link function for this model is the _logit_ function[^logit]:
$$
g(p) = \log \left(\frac{p}{1-p}\right) .
$$

Using this, we get the relationship:
```math
\log \left(\frac{p_i}{1-p_i}\right) = \mathbf{x}_i^\top \mathbb{\beta},
```

and the corresponding mean function is the _logistic_ function[^logistic] $\sigma(\eta) = (1 + e^{-\eta})^{-1}$, that is:
```math
    p_i = \sigma(\mathbf{x}_i^\top \boldsymbol{\beta}) = \frac{1}{1 + e^{-\mathbf{x}_i^\top \boldsymbol{\beta}}}
```

The _logit_ link function is the default used for binary response is implemented as the `LogitLink` object, ans


### Probit Regression: Bernoulli GLM with the Probit link function

Probit regression uses the cumulative distribution function of the standard normal distribution as the _mean_ function.
Let $\Phi(\eta)$ denote the standard normal CDF, then the link function is $\Phi^{-1}(\cdot)$, that is:
```math
\Phi^{-1}(p_i) = \mathbf{x}_i^\top \boldsymbol{\beta},
```
and the mean function is simply:
```math
p_i = \Phi(\mathbf{x}_i^\top \boldsymbol{\beta})
```

The probit link is implemented as the `ProbitLink` type and can be performed using:
```julia
mod = glm(@formula(...), data, Bernoulli(), ProbitLink())
```

### Poisson Regression for count data

When we have count data, i.e. the observed $y_i$'s are non-negative integers we can use the following model: 
$$
    ( Y_i\ \vert\ \mathbf{X}_i = mathbf{x}_i) \sim operatorname{Poisson}(\lambda_i).
$$

Since $\mathb{E}(Y_i\ \vert\ mathbf{x}_i) = \lambda_i$ this can be formulated as a GLM with _log_ as the link function:
```math
\log(\lambda_i) = \mathbf{x}_i^\top \boldsymbol{\beta} .
```
The corresponding mean function is:
```math
\lambda_i = e^{\mathbf{x}_i^\top \boldsymbol{\beta} } .
```
This can be implemented as:
```julia
mod = glm(@formula(...), data, Poisson(), LogLink())
```


[^logit]: https://en.wikipedia.org/wiki/Logit
[^logistic]: https://en.wikipedia.org/wiki/Logistic_function