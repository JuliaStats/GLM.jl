module StatsPlotsExt
    using GLM
    using Statistics
    using StatsPlots
    using RecipesBase
    using Distributions
    using GLM: leverage
    import GLM: cooksleverageplot, cooksleverageplot!
    import GLM: scalelocationplot, scalelocationplot!
    import GLM: residualplot, residualplot!
    import GLM: residualsleverageplot, residualsleverageplot!
    import StatsPlots: QQPlot, QQNorm
    import StatsPlots: qqplot, qqplot!, qqnorm, qqnorm!


    function standardized_residuals(obj::LinearModel)
        r = residuals(obj)
        h = leverage(obj)
        return r ./(std(r) .* sqrt.(1 .- h))
    end

    @recipe function f(l::LinearModel)

    end

    @userplot struct ResidualPlot{T<:Tuple{LinearModel}}
        args::T
    end

    @recipe function f(rp::ResidualPlot; )
        xlabel --> "Fitted values"
        ylabel --> "Residuals"
        title --> "Residuals vs Fitted"
        label --> ""

        r = residuals(rp.args[1])
        y = predict(rp.args[1])

        @series begin
            seriestype := :scatter
            y, r
        end

        @series begin
            seriestype := :hline
            linecolor := :black
            linestyle := :dash
            linewidth := 0.5
            label := ""
            [0.0]
        end
        nothing
    end

    @userplot struct ScaleLocationPlot{T<:Tuple}
        args::T
    end

    @recipe function f(slp::ScaleLocationPlot)
        xlabel --> "Fitted values"
        ylabel --> "√|standardized residuals|"
        title --> "Scale-Location"
        label --> ""

        r = standardized_residuals(slp.args[1])
        y = predict(slp.args[1])
        @series begin
            seriestype := :scatter
            y, (sqrt ∘ abs).(r)
        end
        nothing
    end

    #=
    @userplot struct QuantileQuantilePlot{T<:Tuple{LinearModel}}
        args::T
    end

    @recipe function f(qqp::QuantileQuantilePlot)
        xlabel --> "Theoretical Quantiles"
        ylabel --> "Standardized residuals"
        title --> "Q-Q Residuals"
        label --> ""

        r = residuals(qqp.args[1])
        @series begin
            seriestype := :qqnorm
            linestyle := :dash
            linecolor := :black
            linewidth := 0.5
            r
        end
    end
    =#
    @recipe function f(::QQPlot, obj::LinearModel)
        xlabel --> "Theoretical Quantiles"
        ylabel --> "Standardized Residuals"
        title --> "Q-Q Residuals"
        label --> ""

        linestyle --> :dash
        linecolor --> :gray
        linewidth --> 0.5

        QQPlot(Normal, standardized_residuals(obj))
    end

    # This feels hacky but it works
   qqplot(l::LinearModel, D=Normal;
        xlabel = "Theoretical Quantiles",
        ylabel = "Standardized Residuals",
        title = "Q-Q Residuals",
        label = "",
        linestyle = :dash,
        linecolor = :gray,
        linewidth = 0.5,
        kw...
    ) = qqplot(D, standardized_residuals(l); lsty=linestyle, lcol=linecolor, lw=linewidth, xlabel=xlabel,ylabel=ylabel,title=title,label=label, kw...)

    qqplot!(p, l::LinearModel, D=Normal; kw...) = qqplot!(p, D, standardized_residuals(l); kw...)

    qqnorm(l::LinearModel; kw...) = qqnorm(standardized_residuals(l); kw...)
    qqnorm!(p, l::LinearModel; kw...) = qqnorm!(p, standardized_residuals(l), kw...)


    @userplot struct ResidualsLeveragePlot{T<:Tuple{LinearModel}}
        args::T
    end

    @recipe function f(rlp::ResidualsLeveragePlot; cook_levels = [0.5, 1.0])
        xlabel --> "Leverage"
        ylabel --> "Standardized residuals"
        title --> "Residuals vs Leverage"
        label --> ""

        r = standardized_residuals(rlp.args[1])
        h = leverage(rlp.args[1])
        k = dof(rlp.args[1]) - 1

        ymax = maximum(abs.(r))
        ylims --> (-1.1*ymax, 1.1*ymax)

        @series begin
            seriestype := :scatter
            h, r
        end

        @series begin
            seriestype := :hline
            linecolor := :black
            linestyle := :dash
            linewidth := 0.5
            label := ""
            [0.0]
        end
        @series begin
            seriestype := :vline
            linecolor := :black
            linestyle := :dash
            linewidth := 0.5
            [0.0]
        end
        if !isempty(cook_levels)
            cookfun = (h,D,k) -> sqrt(D * k * (1-h) / h)
            xmin, xmax = extrema(h)
            xs = LinRange(xmin, 1.1*xmax, 50)
            for D in cook_levels
                @series begin
                    seriestype := :path
                    linecolor := :gray
                    linestyle := :dash
                    annotations := [(xs[end], cookfun(xs[end],D,k), ("$D", 8, :gray))]
                    xs, cookfun.(xs,D,k)
                end

                @series begin
                    seriestype := :scatter
                    markeralpha := 0.0
                    series_annotation := [("$D", 9, :gray)]
                    [1.02 * xs[end]], [cookfun(xs[end],D,k)]
                end

                @series begin
                    seriestype := :path
                    linecolor := :gray
                    linestyle := :dash
                xs, -cookfun.(xs,D,k)
                end
            end
        end


    end

    @userplot struct CooksLeveragePlot{T<:Tuple{LinearModel}}
        args::T
    end

    @recipe function f(clp::CooksLeveragePlot)
        xlabel --> "Leverage"
        ylabel --> "Cook's distance"
        title --> "Cook's distance vs Leverage"
        label --> ""


        h = leverage(clp.args[1])
        cd = cooksdistance(clp.args[1])
        @series begin
            seriestype := :scatter
            h, cd
        end
    end
end
