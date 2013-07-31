const template = Formula(:(~ foo))      # for evaluating the lhs of r.e. terms

function lmer(f::Formula, fr::AbstractDataFrame)
    mf = ModelFrame(f, fr); df = mf.df; n = size(df,1)

    ## extract random-effects terms and check there is at least one
    re = filter(x->Meta.isexpr(x,:call) && x.args[1] == :|, mf.terms.terms)
    k = length(re); k > 0 || error("Formula $f has no random-effects terms")

    ## reorder terms by non-increasing number of levels
    gf = PooledDataVector[df[t.args[3]] for t in re];  # grouping factors
    p = sortperm(Int[length(f.pool) for f in gf];rev=true)
    re = re[p]; gf = gf[p]

    ## create and fill vectors of matrices from the random-effects terms
    u = Array(Matrix{Float64},k); Xs = similar(u); lambda = similar(u)
    rowval = Array(Matrix{Int},k); inds = Array(Any,k); offset = 0
    for i in 1:k                    # iterate over random-effects terms
        t = re[i]
        if t.args[2] == 1 Xs[i] = ones(n,1); p = 1; lambda[i] = ones(1,1)
        else
            Xs[i] = (template.rhs=t.args[2]; ModelMatrix(ModelFrame(template, df))).m
            p = size(Xs[i],2); lambda[i] = eye(p)
        end
        l = length(gf[i].pool); u[i] = zeros(p,l); nu = p*l; ii = gf[i].refs
        inds[i] = ii; rowval[i] = (reshape(1:nu,(p,l)) + offset)[:,ii]
        offset += nu
    end
    Ti = Int; if offset < typemax(Int32) Ti = Int32 end ## use 32-bit ints if possible

    ## create the LMMGeneral object
    X = ModelMatrix(mf); p = size(X.m,2); nz = hcat(Xs...)'; rv = vcat(rowval...)
    LambdatZt = CholmodSparse!(convert(Vector{Ti}, [1:size(nz,1):length(nz)+1]),
                               convert(Vector{Ti}, vec(rv)),
                               vec(nz), offset, n, 0)
    y = float(vector(model_response(mf)))
    L = cholfact(LambdatZt,1.,true); pp = invperm(L.Perm + one(Ti))
    rowvalperm = Ti[pp[rv[i,j]] for i in 1:size(rv,1), j in 1:size(rv,2)]

    LMMGeneral(cholfact(LambdatZt,1.,true),LambdatZt,cholfact(eye(p)),
               X,Xs,X.m\y,inds,lambda,zeros(n),vec(rowvalperm),u,y,false,false)
end
lmer(ex::Expr, fr::AbstractDataFrame) = lmer(Formula(ex), fr)

## fit(m) -> m Optimization the objective using BOBYQA from the NLopt package
function fit(m::LinearMixedModel, verbose=false)
    if !isfit(m)
        k = length(theta(m))
        opt = Opt(:LN_BOBYQA, k)
        ftol_abs!(opt, 1e-6)    # criterion on deviance changes
        xtol_abs!(opt, 1e-6)    # criterion on all parameter value changes
        lower_bounds!(opt, lower(m))
        function obj(x::Vector{Float64}, g::Vector{Float64})
            length(g) == 0 || error("gradient evaluations are not provided")
            objective(solve!(theta!(m,x),true))
        end
        if verbose
            count = 0
            function vobj(x::Vector{Float64}, g::Vector{Float64})
                if length(g) > 0 error("gradient evaluations are not provided") end
                count += 1
                val = obj(x, g)
                println("f_$count: $val, $x")
                val
            end
            min_objective!(opt, vobj)
        else
            min_objective!(opt, obj)
        end
        fmin, xmin, ret = optimize(opt, theta(m))
        if verbose println(ret) end
        m.fit = true
    end
    m
end
