GLM.jl --- Generalized linear models and others
=====================================================

.. toctree::
   :maxdepth: 2

.. highlight:: julia

.. .. module:: GLM.jl
   :synopsis: Fit and analyze mixed-effects models

MixedEffects.jl provides functions and methods to fit `mixed-effects
models <http://en.wikipedia.org/wiki/Mixed_model>`__ using a
specification similar to that of the `lme4
<https://github.com/lme4/lme4>`__ package for `R
<http://www.R-project.org>`__.  Currently the linear mixed models
(LMMs) are implemented.

-------
Example
-------

The :func:`lmm()` function creates a linear mixed model
representation that inherits from :class:`LinearMixedModel`.  The
:class:`LMMGeneral` type can represent any LMM expressed in the
formula language.  Other types are used for better performance in
special cases::

    julia> using GLM, RDatasets

    julia> ds = data("lme4", "Dyestuff");

    julia> dump(ds)
    DataFrame  30 observations of 2 variables
      Batch: PooledDataArray{ASCIIString,Uint8,1}(30) ["A","A","A","A"]
      Yield: DataArray{Float64,1}(30) [1545.0,1440.0,1440.0,1520.0]

    julia> m = lmm(:(Yield ~ 1|Batch), ds);

    julia> typeof(m)
    LMMGeneral{Int32}

    julia> fit(m, true);
    f_1: 327.7670216246145, [1.0]
    f_2: 331.0361932224437, [1.75]
    f_3: 330.6458314144857, [0.25]
    f_4: 327.69511270610866, [0.9761896354668361]
    f_5: 327.56630914532184, [0.9285689064005083]
    f_6: 327.3825965130752, [0.8333274482678525]
    f_7: 327.3531545408492, [0.8071883308459398]
    f_8: 327.34662982410276, [0.7996883308459398]
    f_9: 327.34100192001785, [0.7921883308459399]
    f_10: 327.33252535370985, [0.7771883308459397]
    f_11: 327.32733056112147, [0.7471883308459397]
    f_12: 327.3286190977697, [0.7396883308459398]
    f_13: 327.32706023603697, [0.7527765100471926]
    f_14: 327.3270681545395, [0.7535265100471926]
    f_15: 327.3270598812218, [0.7525837539477753]
    FTOL_REACHED

    julia> m
    Linear mixed model fit by maximum likelihood
     logLik: -163.6635299406109, deviance: 327.3270598812218

      Variance components:
	Std. deviation scale: [37.26047449632836]
	Variance scale: [1388.342959691536]
      Number of obs: 30; levels of grouping factors: [6]

      Fixed-effects parameters:
	    Estimate Std.Error z value
    [1,]      1527.5   17.6946 86.3258

------------
Constructors
------------

.. function:: lmm(f, fr)

   Create the representation for a linear mixed-effects model with
   formula ``f`` evaluated in the :type:`DataFrame` ``fr``.  The
   primary method is for ``f`` of type :type:`Formula` but more
   commonly ``f`` is an expression (:type:`Expr`) as in the example
   above.

-------
Setters
-------

These setters or mutating functions are defined for ``m`` of type
:type:`LMMGeneral`.  By convention their names end in ``!``.  The
:func:`fit` function is an exception, because the name was
already established in the ``Distributions`` package.

.. function:: fit(m, verbose=false) -> m

   Fit the parameters of the model by maximum likelihood or by the REML criterion.

.. function:: reml!(m, v=true]) -> m

   Set the REML flag in ``m`` to ``v``.

.. function:: solve!(m, ubeta=false) -> m

   Update the random-effects values (and the fixed-effects, when
   ``ubeta`` is ``true``) by solving the penalized least squares (PLS)
   problem.

.. function:: theta!(m, th) -> m

   Set a new value of the variance-component parameter and update the
   sparse Cholesky factor.

----------
Extractors
----------

These extractors are defined for ``m`` of type
:type:`LMMGeneral`.

.. function:: cholfact(m,RX=true) -> Cholesky{Float64} or CholmodFactor{Float64}

   The Cholesky factor, ``RX``, of the downdated X'X or the sparse
   Cholesky factor, ``L``, of the random-effects model matrix in the U
   scale.  These are returned as references and should not be modified.

.. function:: coef(m) -> Vector{Float64}

   A synonym for :func:`fixef`

.. function:: coeftable(m) -> DataFrame

   A dataframe with the current fixed-effects parameter vector, the
   standard errors and their ratio.

.. function:: cor(m) -> Vector{Matrix{Float64}}

   Vector of correlation matrices for the random effects

.. function:: deviance(m) -> Float64

   Value of the deviance (returns ``NaN`` if :func:`isfit` is ``false`` or
   :func:`isreml` is ``true``).

.. function:: fixef(m) -> Vector{Float64}

   Fixed-effects parameter vector

.. function:: grplevels(m) -> Vector{Int}

   Vector of number of levels in random-effect terms

.. function:: linpred(m, minusy=true) -> Vector{Float64}

   The linear predictor vector or the negative residual vector

.. function:: lower(m) -> Vector{Float64}

   Vector of lower bounds on the variance-component parameters

.. function:: objective(m) -> Float64

   Value of the profiled deviance or REML criterion at current parameter values

.. function:: pwrss(m) -> Float64

   The penalized, weighted residual sum of squares.

.. function:: ranef(m, uscale=false) -> Vector{Matrix{Float64}}

   Vector of matrices of random effects on the original scale or on the U scale

.. function:: scale(m, sqr=false) -> Float64

   Estimate, ``s``, of the residual scale parameter or its square.

.. function:: std(m) -> Vector{Float64}

   Estimated standard deviations of random effects.

.. function:: stderr(m) -> Vector{Float64}

   Standard errors of the fixed-effects parameters

.. function:: theta(m) -> Vector{Float64}

   Vector of variance-component parameters

.. function:: vcov(m) -> Matrix{Float64}

   Estimated variance-covariance matrix of the fixed-effects parameters

----------
Predicates
----------

The following predicates (functions that return boolean values,
:type:`Bool`) are defined for `m` of type :type:`LMMGeneral`

.. function:: isfit(m)

   Has the model been fit?

.. function:: isreml(m)

   Is the model to be fit by REML?

.. function:: isscalar(m)

   Are all the random-effects terms scalar?

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
