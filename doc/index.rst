GLM.jl --- Generalized linear models and others
=====================================================

.. toctree::
   :maxdepth: 2

.. highlight:: julia

.. .. module:: GLM.jl
   :synopsis: Fit and analyze linear and generalized linear models

The `GLM` package for `Julia <http://julialang.org>`__ provides
functions and methods to fit `linear regression models
<http://en.wikipedia.org/wiki/Linear_models>`__ and `generalized
linear models
<http://en.wikipedia.org/wiki/Generalized_linear_model>`__ using a
specification similar to that of for `R <http://www.R-project.org>`__.

-------
Example
-------

The :func:`fit(LinearModel, ...)` method creates a linear model
representation that inherits from :class:`LinearModel`.  The abstract
:class:`LinPredModel` type includes both linear and generalized linear models.

    julia> using GLM, DataFrames, RDatasets

    julia> form = dataset("datasets", "Formaldehyde")
    6x2 DataFrame
    |-------|------|--------|
    | Row # | Carb | OptDen |
    | 1     | 0.1  | 0.086  |
    | 2     | 0.3  | 0.269  |
    | 3     | 0.5  | 0.446  |
    | 4     | 0.6  | 0.538  |
    | 5     | 0.7  | 0.626  |
    | 6     | 0.9  | 0.782  |

    julia> lm1 = lm(OptDen ~ Carb, form)
    Formula: OptDen ~ Carb

    Coefficients:
		   Estimate  Std.Error  t value Pr(>|t|)
    (Intercept)  0.00508571 0.00783368 0.649211   0.5516
    Carb           0.876286  0.0135345  64.7444   3.4e-7

    julia> dobson = DataFrame(counts=[18.,17,15,20,10,20,25,13,12], outcome=gl(3,1,9), treatment=gl(3,3))
    9x3 DataFrame
    |-------|--------|---------|-----------|
    | Row # | counts | outcome | treatment |
    | 1     | 18.0   | 1       | 1         |
    | 2     | 17.0   | 2       | 1         |
    | 3     | 15.0   | 3       | 1         |
    | 4     | 20.0   | 1       | 2         |
    | 5     | 10.0   | 2       | 2         |
    | 6     | 20.0   | 3       | 2         |
    | 7     | 25.0   | 1       | 3         |
    | 8     | 13.0   | 2       | 3         |
    | 9     | 12.0   | 3       | 3         |

    julia> gm1 = glm(counts ~ outcome + treatment, dobson, Poisson())
    Formula: counts ~ :(+(outcome,treatment))

    Coefficients:
		       Estimate Std.Error      z value Pr(>|z|)
    (Intercept)         3.04452  0.170899      17.8148  < eps()
    outcome - 2       -0.454255  0.202171     -2.24689   0.0246
    outcome - 3       -0.292987  0.192742      -1.5201   0.1285
    treatment - 2   5.36273e-16       0.2  2.68137e-15      1.0
    treatment - 3  -5.07534e-17       0.2 -2.53767e-16      1.0

------------
Constructors
------------

.. function:: fit(LinearModel, X, y)

   Create the representation for a linear regression model with design
   matrix ``X`` and response vector ``y``. When DataFrames is imported,
   ``X`` and ``y`` may also be a model formula and :type:`DataFrame`
   respectively.


.. function:: lm(X, y)

   Alias for ``fit(LinearModel, X, y)``.


.. function:: fit(GeneralizedLinearModel, X, y, d[, l])

   Create the representation for a generalized linear model with design
   matrix ``X``, response vector ``y``, distribution ``d`` and,
   optionally, link ``l``.  When DataFrames is imported, ``X`` and ``y``
   may also be a model formula and :type:`DataFrame` respectively. If
   ``l`` is omitted the canonical link for ``d`` is used.


.. function:: glm(X, y, d[, l])

   Alias for ``fit(GeneralizedLinearModel, X, y, d[, l])``.


----------
Extractors
----------

These extractors are defined for ``m`` of type
:type:`LinPredModel`.

.. function:: coef(m) -> Vector{Float64}

   Coefficient estimates

.. function:: coeftable(m) -> DataFrame

   A dataframe with the current fixed-effects parameter vector, the
   standard errors, their ratio and the p-value for the ratio.

.. function:: confint(m[, level]) -> Matrix{Float64}

   A matrix of the lower and upper end-points of the (marginal)
   confidence intervals on the coefficients.  The confidence level,
   ``level``, defaults to 0.95.

.. function:: scale(m, sqr=false) -> Float64

   Estimate, ``s``, of the residual scale parameter or its square.

.. function:: stderr(m) -> Vector{Float64}

   Standard errors of the fixed-effects parameters

.. function:: vcov(m) -> Matrix{Float64}

   Estimated variance-covariance matrix of the fixed-effects parameters

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
