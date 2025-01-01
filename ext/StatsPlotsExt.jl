module StatsPlotsExt
    using GLM
    using Statistics
    using StatsPlots
    using RecipesBase
    using Distributions
    using GLM: leverage, standardized_residuals
    using RecipesBase: recipetype
    import GLM: cooksleverageplot, cooksleverageplot!
    import GLM: scalelocationplot, scalelocationplot!
    import GLM: residualplot, residualplot!
    import GLM: residualsleverageplot, residualsleverageplot!
    import StatsPlots: qqplot, qqplot!, qqnorm, qqnorm!
    import GLM: lmplot


    function lmplot(obj::LinearModel; kw...)
        return plot(
            residualplot(obj),
            qqplot(obj),
            scalelocationplot(obj),
            residualsleverageplot(obj);
            layout = (2,2),
            kw...
        )
    end

    @userplot struct ResidualPlot{T<:Tuple{LinearModel}}
        args::T
    end

    @recipe function f(rp::ResidualPlot;
        axislines = true,
        axislinestyle = :dot,
        axislinecolor = :black,
        axislinewidth = 0.5
        )
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
        if axislines
            @series begin
                seriestype := :hline
                linecolor := axislinecolor
                linestyle := axislinestyle
                linewidth := axislinewidth
                label := ""
                [0.0]
            end
        end
        nothing
    end

    @userplot struct ScaleLocationPlot{T<:Tuple{LinearModel}}
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



    function qqplot(l::LinearModel, D=Normal;
        xlabel = "Theoretical Quantiles",
        ylabel = "Standardized Residuals",
        title = "Q-Q Residuals",
        label = "",
        linestyle = :dash,
        linecolor = :black,
        linewidth = 0.5,
        kw...
    )
        qqplot(D, standardized_residuals(l);
        linestyle=linestyle,
        linecolor=linecolor,
        linewidth=linewidth,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        label=label,
        kw...)
    end

    function qqplot!(p, l::LinearModel, D=Normal;
        xlabel = "Theoretical Quantiles",
        ylabel = "Standardized Residuals",
        title = "Q-Q Residuals",
        label = "",
        linestyle = :dash,
        linecolor = :black,
        linewidth = 0.5,
        kw...
        )
        qqplot!(p, D, standardized_residuals(l);
        linestyle=linestyle,
        linecolor=linecolor,
        linewidth=linewidth,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        label=label,
        kw...)
    end

    function qqnorm(l::LinearModel;
        xlabel = "Theoretical Quantiles",
        ylabel = "Standardized Residuals",
        title = "Q-Q Residuals",
        label = "",
        linestyle = :dash,
        linecolor = :black,
        linewidth = 0.5,
        kw...
        )
        qqnorm(standardized_residuals(l);
        linestyle=linestyle,
        linecolor=linecolor,
        linewidth=linewidth,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        label=label,
        kw...)
    end

    function qqnorm!(p, l::LinearModel;
        xlabel = "Theoretical Quantiles",
        ylabel = "Standardized Residuals",
        title = "Q-Q Residuals",
        label = "",
        linestyle = :dash,
        linecolor = :black,
        linewidth = 0.5,
        kw...)
        qqnorm!(p, standardized_residuals(l),
        linestyle=linestyle,
        linecolor=linecolor,
        linewidth=linewidth,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        label=label,
        kw...)
    end

    @userplot struct ResidualsLeveragePlot{T<:Tuple{LinearModel}}
        args::T
    end

    @recipe function f(rlp::ResidualsLeveragePlot;
        axislines = true,
        axislinestyle = :dot,
        axislinecolor = :gray,
        axislinewidth = 1,
        cookslevels = [0.5, 1.0],
        cookslinecolor = :gray,
        cookslinestyle = :dash,
        cookslinewidth = 1
        )
        xlabel --> "Leverage"
        ylabel --> "Standardized residuals"
        title --> "Residuals vs Leverage"
        label --> ""

        r = standardized_residuals(rlp.args[1])
        h = leverage(rlp.args[1])
        k = dof(rlp.args[1]) - 1

        ymax = maximum(abs.(r))
        ylims --> (-1.1*ymax, 1.1*ymax)
        xlims --> (0.0, :auto)

        @series begin
            seriestype := :scatter
            h, r
        end
        if axislines
            @series begin
                seriestype := :hline
                linecolor := axislinecolor
                linestyle := axislinestyle
                linewidth := axislinewidth
                label := ""
                [0.0]
            end
            @series begin
                seriestype := :vline
                linecolor := axislinecolor
                linestyle := axislinestyle
                linewidth := axislinewidth
                [0.0]
            end
        end
        if !isempty(cookslevels)
            cookfun = (h,D,k) -> sqrt(D * k * (1-h) / h)
            xmin, xmax = extrema(h)
            xs = LinRange(xmin, 1.2*xmax, 50)
            for D in cook_levels
                @series begin
                    seriestype := :path
                    linecolor := cookslinecolor
                    linestyle := cookslinestyle
                    linewidth := cookslinewidth
                    annotations := [(xs[end], cookfun(xs[end],D,k), ("$D", 8, :gray))]
                    xs, cookfun.(xs,D,k)
                end

                @series begin
                    seriestype := :scatter
                    markeralpha := 0.0
                    series_annotation := [("$D", 8, :gray)]
                    [1.02 * xs[end]], [cookfun(xs[end],D,k)]
                end
                @series begin
                    seriestype := :scatter
                    markeralpha := 0.0
                    series_annotation := [("$D", 8, :gray)]
                    [1.02 * xs[end]], [-cookfun(xs[end],D,k)]
                end

                @series begin
                    seriestype := :path
                    linecolor := cookslinecolor
                    linestyle := cookslinestyle
                    linewidth := cookslinewidth
                    xs, -cookfun.(xs,D,k)
                end
            end
            begin
                seriestype := :path
                linecolor := cookslinecolor
                linestyle := cookslinestyle
                linewidth := cookslinewidth
                label := "Cook's Distance"
                [-1,-1],[-1,-1]
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
