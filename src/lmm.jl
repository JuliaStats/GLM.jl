const template = Formula(:(~ foo))      # for evaluating the lhs of r.e. terms

function lmm(f::Formula, fr::AbstractDataFrame; dofit=true)
    mf = ModelFrame(f, fr); df = mf.df; n = size(df,1)
    
    ## extract random-effects terms and check there is at least one
    re = filter(x->Meta.isexpr(x,:call) && x.args[1] == :|, mf.terms.terms)
    k = length(re); k > 0 || error("Formula $f has no random-effects terms")
    
    ## reorder terms by non-increasing number of levels
    gf = PooledDataVector[df[t.args[3]] for t in re]  # grouping factors
    p = sortperm(Int[length(f.pool) for f in gf]; rev=true)
    re = re[p]; gf = gf[p]

    ## create and fill vectors of matrices from the random-effects terms
    u = Array(Matrix{Float64},k); Xs = similar(u); lambda = similar(u)
    rowval = Array(Matrix{Int},k); inds = Array(Any,k); offset = 0; scalar = true
    for i in 1:k                    # iterate over random-effects terms
        t = re[i]
        if t.args[2] == 1 Xs[i] = ones(n,1); p = 1; lambda[i] = ones(1,1)
        else
            Xs[i] = (template.rhs=t.args[2]; ModelMatrix(ModelFrame(template, df))).m
            p = size(Xs[i],2); lambda[i] = eye(p)
        end
        if p > 1; scalar = false; end
        l = length(gf[i].pool); u[i] = zeros(p,l); nu = p*l; ii = gf[i].refs
        inds[i] = ii; rowval[i] = (reshape(1:nu,(p,l)) + offset)[:,ii]
        offset += nu
    end
    Ti = Int; if offset < typemax(Int32) Ti = Int32 end ## use 32-bit ints if possible

    X = ModelMatrix(mf); rv = convert(Matrix{Ti},vcat(rowval...))
    y = float64(vector(model_response(mf)))
                                     # create the appropriate type of LMM object
#    m = k == 1 && scalar ? LMMScalar1(X.m', vec(rv), vec(Xs[1]), y) :
    m = LMMGeneral(offset,X,Xs,inds,u,rv,y,lambda)
    println(typeof(m))
    dofit ? fit(m) : m
end
lmm(ex::Expr, fr::AbstractDataFrame) = lmer(Formula(ex), fr)
