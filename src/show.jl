function show(io::Any, obj::LmMod)
    cc = coef(obj)
    se = stderr(obj)
    tt = cc ./ se
    pp = ccdf(FDist(1, df_residual(obj)), tt .* tt)
    @printf("\n%s\n\nCoefficients:\n", obj.fr.formula)
    @printf("         Term    Estimate  Std. Error     t value    Pr(>|t|)\n")
    N = length(cc)
    for i = 1:N
        @printf(" %12s%12.5f%12.5f%12.3f%12.3f %-3s\n",
				obj.mm.model_colnames[i],
                cc[i],
                se[i],
                tt[i],
                pp[i],
                p_value_stars(pp[i]))
    end
    println("---\nSignif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1\n")
    @printf("R-squared: %0.4f\n", 0.0) # TODO: obj.r_squared)
end
repl_show(io::Any, obj::LmMod) = show(io, obj)

function show(io::Any, obj::GlmMod)
    cc = coef(obj)
    se = stderr(obj)
    zz = cc ./ se
    pp = 2.0 * ccdf(Normal(), abs(zz))
    @printf("\n%s\n\nCoefficients:\n", obj.fr.formula)
    @printf("         Term    Estimate  Std. Error     t value    Pr(>|t|)\n")
    N = length(cc)
    for i = 1:N
        @printf(" %12s%12.5f%12.5f%12.3f%12.3f %-3s\n",
                obj.mm.model_colnames[i],
                cc[i],
                se[i],
                zz[i],
                pp[i],
                p_value_stars(pp[i]))
    end
    println("---\nSignif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1\n")
    @printf("R-squared: %0.4f\n", 0.0) # TODO: obj.r_squared)
end
repl_show(io::Any, obj::GlmMod) = show(io, obj)

function p_value_stars(p_value::Float64)
    if p_value < 0.001
        return "***"
    elseif p_value < 0.01
        return "**"
    elseif p_value < 0.05
        return "*"
    elseif p_value < 0.1
        return "."
    else
        return " "
    end
end
