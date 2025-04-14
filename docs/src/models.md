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
The link function maps the mean of the distribution to the linear predictor, that is:
```math
g(\mathbb{E}(Y_i | \mathbf{x}_i)) = \mathbf{x}_i^\top \mathbb{\beta} = \eta_i .
```
The link function must be invertible, and the inverse link function is also sometimes referred to as the mean function as it translates the linear predictor into the mean, that is:
```math
    \mathbb{E}(Y_i | \mathbf{x}_i) = g^{-1}(\eta_i) .
```
Another (perhaps more intuitive) way to see this is that the _mean_ function maps the linear predictor to the space of possible mean values.
For instance, when we have binary response the mean must be in the interval $[0,1]$ however, the linear predictor can take any real value.
The mean function (say logistic function, more on that later) takes the linear predictor and squishes into the $(0, 1)$ interval.

In short, the link function allows us to go from the mean to the associated linear predictor and the mean function allows us to go from the linear predictor to the mean.

In short, a GLM consists of three key components[^GLMwiki]:

1. The linear predictor
2. The distribution family
3. The link function $g(\mu) = \eta$

Different combinations of the family and link define the various GLMs:
```julia
glm(formula, family, link)
```

### Supported Distribtions

`GLM.jl` supports models with following distribution families:

- Normal distribution (`family = Normal()`)
- Bernoulli  (`family = Bernoulli()`)
- Binomial
- Gamma
- Geometric
- Inverse Gaussian
- Negative Binomial
- Poisson





### Supported Link functions

These can be combined with an appropriate link function from the following list:

| Link: Constructor | Link function | Mean function |
|:---------- |:---------- |:------------|
| | $\eta = g(\mu)$ | $\mu = h(\eta) = g^{-1}(\eta)$ |
| Cauchit link: `CauchitLink()` | $\eta = \tan(\pi \times (\mu - \frac{1}{2}))$ | $\mu = \frac{1}{2}+ \frac{\tan^{-1}(\eta)}{\pi}$|
| complimentary log log link: `CloglogLink()` | $\eta = \log(-\log(1 - \mu))$ | $\mu = 1 - \exp(-\exp(\eta))$ |
| Identity link: `IdentityLink()` | $\eta = \mu$ | $\mu = \eta$ |
| Inverse Link (or reciprocal) : `InverseLink()` | $\eta = \frac{1}{\mu}$ | $\mu = \frac{1}{\eta}$ |
| Inverse square link: `InverseSquareLink()` | $\eta = \frac{1}{\mu^2}$ | $\mu = \frac{1}{\eta^2}$ |
| Logit link: `LogitLink()` | $\eta = \log(\frac{\mu}{1 - \mu})$ | $\mu = (1 + e^{-\eta})^{-1}$ |
| Log link: `LogLink()` | $\eta = \log(\mu)$ | $\mu = \exp(\eta)$ |
| Negative Binomial link: `NegativeBinomialLink(θ)` | $\eta = \log(\frac{\mu}{\mu + \theta})$ | $\mu = \frac{\theta \exp(\eta)}{1 - \exp(\eta)}$ |
| Power link: `PowerLink(k)`| $\eta = \mu^{k}$ | $\mu = \eta^{1/k}$ |
| Probit link: `ProbitLink()` | $\eta = \Phi^{-1}(\mu)$ | $\mu = \Phi(\eta)$ |
| Square root Link: `SqrtLink()` |$\eta = \sqrt{\mu}$ | $\mu = \sqrt{\eta}$ |

Note that not all combinations of distribution and link are appropriate.
The following table summarizes the combinations


| Link\Dist | Normal |  Bern. |  Binom. |  Gamma |  Geom. |  Inv-Gaussian |  Neg-Binom |  Pois. |
|:--|:------:|:----------:|:---------:|:------:|:----------:|:-----------------:|:--------:|:------:|
| Identity | `lm` | ❌ | ❌ | ? | ? | ? | ❌ | ❌ |
| Cauchit | ? | ✓ | ✓ | ? | ? | ? | ? | ? |
| Cloglog  | ? | ✓ | ✓ | ? | ? | ? | ? | ? |
| Inverse | ? | ? | ? | ✅ | ? | ? | ? | ? |
| Logit | ? | ✅ | ✓ | ? | ? | ? | ? | ? |
| Log | ? | ? | ? | ? | ? | ? | ✓ | ✅ |
| NegBinomial | ? | ? | ? | ? | ? | ? | ✅ |  ✓ |

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
```math
    ( Y_i\ \vert\ \mathbf{X}_i = \mathbf{x}_i) \sim \operatorname{Poisson}(\lambda_i).
```

Since $\mathbb{E}(Y_i\ \vert\ \mathbf{x}_i) = \lambda_i$ this can be formulated as a GLM with _log_ as the link function:
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


### Binomial Regression

**Note:** When performing binomial regression, the response variable must be a proportion, i.e. $0 \leq y_i \leq 1$.
If the data contains the number of successes and the number of trials, first generate the success proportion, and use that as the response variable.

We want to model the number of successes from $n_i$ trials using a Binomial distribution as:
```math
    Y_i \sim \operatorname{Binomial}(n_i, p_i) .
```
However, we work with the proportion of successes and model the mean parameter $p_i$ which is:
```math
p_i = \mathbb{E}\left(\frac{Y_i}{n_i} \right) ,
```
using the link function:
```math
    \log\left(\frac{p_i}{1 - p_i}\right) = \mathbf{x}_i^\top \boldsymbol{\beta} ,
```
and the corresponding mean function is the _logistic_ function $p_i = \sigma(\mathbf{x}_i^\top \boldsymbol{\beta})$.

### Geometric GLM

We wish to model count responses $Y_i \in \{0, 1, 2, ... \}$ using the Geometric distribution, that is:
```math
    Y_i \sim \operatorname{Geometric}(p_i),
```
which has the mean $\mu_i = \frac{1 - p_i}{p_i}$.
The canonical link function for Geometric GLM is the _log_ link:
```math
    log(\mu_i) =\mathbf{x}_i^\top \boldsymbol{\beta}.
```



## Negative Binomial Models

There are many parameterizations of the negative binomial distribution, but we use the following:
```math
p(Y = y) = \frac{\Gamma(y + \theta)}{ \Gamma(\theta) \times y!} \times \frac{\theta^\theta \mu^y}{(\mu + \theta)^{(y + \theta)}} ,
```
where $\mu$ is the mean of the distribution and $\theta$ is the number of successes when the experiment is stopped.
This parameterization can easily be derived by noting that $\mu = \frac{\theta(1 - p)}{p}$, and plugging in the corresponding value of $p$ in the more commonly used parameterization detailed on [Wikipedia](https://en.wikipedia.org/wiki/Negative_binomial_distribution).
Also note that we use the $\Gamma$ function instead of factorial here, because we will allow the $\theta$ parameter to take noninteger positive values.

We will model the mean as a function of the predictor variables.
The canonical link function for this distribution is the negative binomial link function that is:
```math
    \log \left(\frac{\mu_i}{\mu_i + \theta}\right) = \mathbf{x}_i^{\top} \boldsymbol{\beta}
```
but the log-link function is frequently used:
```math
    \log(\mu_i) = \mathbf{x}_i^{\top} \boldsymbol{\beta}
```

Notice that the canonical link function takes $\theta$ as an argument.
This is because the negative binomial distribution belongs to the exponential family only if the number of successes parameter ($\theta$) is known.
As a result, a model that attempts to estimate both $\theta$ and $\boldsymbol{\beta}$ simultaneously is technically not a GLM.
However, if the $\theta$ parameter is specified beforehand the model is a GLM.

This package supports both versions of the model, that is (1) the GLM version with the $\theta$ parameter is assumed to be known, and (2) the more general model where $\theta$ is estimated along with the $\beta$ parameter.

## Negative Binomial Model with known $\theta$

This model can be fit as:
```julia
glm(frm, dt, NegativeBinomial(θ))
```
or by supplying the log-link:
```julia
glm(frm, dt, NegativeBinomial(θ), LogLink())
```

## Negative binomial Model with Simultaneous Estimation of $\theta$

While this is technically not a GLM, the package does support this version using the `negbin` function which can be called as follows:
```julia
negbin(frm, dt)
```
and optionally using a link function (usually the log-link) as:
```julia
negbin(frm, dt, link)
```

This function estimates both the coefficient vector $\boldsymbol{\beta}$ and the parameter $\theta$.
This is done in an iterative procedure in which the coefficient vector is estimated using a GLM assuming $\theta$ is fixed (and set to its previous estimate), and the $\theta$ is estimated by maximizing the log-likelihood assuming the coefficient vector is fixed (set to its previous estimate).
The first step in each iteration internally calls `glm`.


[^logit]: https://en.wikipedia.org/wiki/Logit
[^logistic]: https://en.wikipedia.org/wiki/Logistic_function
[^GLMwiki]: https://en.wikipedia.org/wiki/Generalized_linear_model#Model_components
