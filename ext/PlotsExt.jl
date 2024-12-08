module PlotsExt
    using Plots
    using RecipesBase

    @userplot struct LMPlot{T<:Tuple{LinearModel}}
        args::T
    end

    @recipe function f(lmp::LMPlot)

    end

    @userplot struct ResidualPlot{T<:Tuple}
        args::T
    end

    @recipe function f(rp::ResidualPlot)
        xlabel --> "Fitted values"
        ylabel --> "Residuals"
        title --> "Residuals vs Fitted"
    end

    @userplot struct ScaleLocationPlot{T<:Tuple}
        args::T
    end

    @recipe function f(slp::ScaleLocationPlot)
        xlabel --> "Fitted values"
        ylabel --> L"\sqrt{|\text{standardized residuals}|}"
        title --> "Scale-Location"
    end

    @userplot struct QQPlot{T<:Tuple}
        args::T
    end

    @recipe function f(qqp::QQPlot; qqline=true)
        xlabel --> "Theoretical Quantiles"
        ylabel --> "Standardized residuals"
        title --> "Q-Q Residuals"

        r = residuals(qqp.args[1])
    end


    @userplot struct ResidualsLeveragePlot{T<:Tuple{LinearModel}}
        args::T
    end

    @recipe function f(rlp::ResidualsLeveragePlot)
        xlabel --> "Leverage"
        ylabel --> "Standardized residuals"
        title --> "Residuals vs Leverage"

        r = residuals(rlp.args[1])
    end

    @userplot struct CooksLeveragePlot{T<:Tuple{LinearModel}}
        args::T
    end

    @recipe function f(clp::CooksLeveragePlot)
        xlabel --> L"\text{Leverage } h_{ii}"
        ylabel --> "Cook's distance"
        title --> L"\text{Cook's distance vs Leverage }*h_{ii}/(1-h_{ii})"
    end
end
