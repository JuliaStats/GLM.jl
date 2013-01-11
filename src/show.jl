function show(io::Any, obj::LmMod)
    @printf("\n%s\n\nCoefficients:\n", obj.fr.formula)
    @printf("         Term    Estimate  Std. Error     t value    Pr(>|t|)\n")
    N = length(obj.pp.beta0)
    for i = 1:N
        @printf(" %12s%12.5f%12.5f%12.3f%12.3f %-3s\n",
				obj.mm.model_colnames[i],
                obj.pp.beta0[i],
                obj.pp.beta0[i], # TODO: obj.std_errors[i],
                obj.pp.beta0[i], # TODO: obj.t_statistics[i],
                obj.pp.beta0[i], # TODO: obj.p_values[i],
                p_value_stars(obj.pp.beta0[i])) # TODO: p_value_stars(obj.p_values[i]))
    end
    println("---\nSignif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1\n")
    @printf("R-squared: %0.4f\n", 0.0) # TODO: obj.r_squared)
end
repl_show(io::Any, obj::LmMod) = show(io, obj)

# function show(io::Any, obj::TestResult)
#     if length(obj.df) == 1
#         @printf("%8s   Pr(>%s)\n%8.3f%8.3f", 
#                 @sprintf("%s(%d)", obj.distribution, obj.df[1]),
#                 obj.distribution,
#                 obj.test_statistic,
#                 obj.p_value)
#     else
#         @printf("%8s   Pr(>%s)\n%8.3f%8.3f", 
#                 @sprintf("%s(%d,%d)", obj.distribution, obj.df[1], obj.df[2]),
#                 obj.distribution,
#                 obj.test_statistic,
#                 obj.p_value)
#     end
# end

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
