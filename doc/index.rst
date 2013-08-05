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

The :func:`lm()` function creates a linear model
representation that inherits from :class:`LmMod`.  The abstract
:class:`LinPredModel` type includes both linear and generalized linear models.

    julia> using GLM, RDatasets

    julia> form = data("datasets", "Formaldehyde")
    6x2 DataFrame:
	    carb optden
    [1,]     0.1  0.086
    [2,]     0.3  0.269
    [3,]     0.5  0.446
    [4,]     0.6  0.538
    [5,]     0.7  0.626
    [6,]     0.9  0.782


    julia> lm1 = lm(:(optden ~ carb), form)

    Formula: optden ~ carb

    Coefficients:

    2x4 DataFrame:
	      Estimate  Std.Error  t value   Pr(>|t|)
    [1,]    0.00508571 0.00783368 0.649211   0.551595
    [2,]      0.876286  0.0135345  64.7444 3.40919e-7

    julia> dobson = DataFrame(counts=[18.,17,15,20,10,20,25,13,12], outcome=gl(3,1,9), treatment=gl(3,3));

    julia> dump(dobson)
    DataFrame  9 observations of 3 variables
      counts: DataArray{Float64,1}(9) [18.0,17.0,15.0,20.0]
      outcome: PooledDataArray{Int64,Uint8,1}(9) [1,2,3,1]
      treatment: PooledDataArray{Int64,Uint8,1}(9) [1,1,1,2]

    julia> gm1 = glm(:(counts ~ outcome + treatment), dobson, Poisson())

    Formula: counts ~ :(+(outcome,treatment))

    Coefficients:

    5x4 DataFrame:
		Estimate Std.Error      z value    Pr(>|z|)
    [1,]         3.04452  0.170899      17.8148 5.42677e-71
    [2,]       -0.454255  0.202171     -2.24689   0.0246471
    [3,]       -0.292987  0.192742      -1.5201    0.128487
    [4,]     5.36273e-16       0.2  2.68137e-15         1.0
    [5,]    -5.07534e-17       0.2 -2.53767e-16         1.0

------------
Constructors
------------

.. function:: lm(f, fr)

   Create the representation for a linear regression model with
   formula ``f`` evaluated in the :type:`DataFrame` ``fr``.  The
   primary method is for ``f`` of type :type:`Formula` but more
   commonly ``f`` is an expression (:type:`Expr`) as in the example
   above.

.. function:: glm(f, fr, d[, l])

   Create the representation for a linear regression model with
   formula ``f`` evaluated in the :type:`DataFrame` ``fr`` with
   distribution ``d`` and, optionally, link ``l``.  The primary method is
   for ``f`` of type :type:`Formula` but more commonly ``f`` is an
   expression (:type:`Expr`) as in the example above.  If ``l`` is
   omitted the canonical link for ``d`` is used.

----------
Extractors
----------

These extractors are defined for ``m`` of type
:type:`LMMGeneral`.

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
