module MakieExt
    using Makie
    using GLM
    using Distributions
    using GLM: leverage, standardized_residuals
    import GLM: cooksleverageplot, cooksleverageplot!
    import GLM: scalelocationplot, scalelocationplot!
    import GLM: residualplot, residualplot!
    import GLM: residualsleverageplot, residualsleverageplot!
    import GLM: lmplot
    import Makie: qqplot, qqplot!


    function lmplot(obj::LinearModel; figkw...)
        fig = Figure(; figkw...)
        ax_1 = Axis(fig[1,1],
            title = "Residuals vs Fitted Values",
            xlabel = "Fitted Values",
            ylabel = "Residuals"
        )
        residualplot!(ax_1, obj)
        ax_2 = Axis(fig[1,2],
            title = "Q-Q Residuals",
            xlabel = "Theoretical Quantiles",
            ylabel = "Standardized Residuals"
        )
        qqplot!(ax_2, obj)
        ax_3 = Axis(fig[2,1],
            title = "Scale-Location",
            xlabel = "Fitted Values",
            ylabel = "√|standardized residuals|"
        )
        scalelocationplot!(ax_3, obj)
        ax_4 = Axis(fig[2,2],
            title = "Residuals vs Leverage",
            xlabel = "Leverage",
            ylabel = "Standardized Residuals"
        )
        ymax = maximum(abs.(standardized_residuals(obj)))
        xmax = maximum(leverage(obj))
        ylims!(ax_4, -1.2*ymax, 1.2*ymax)
        xlims!(ax_4, 0.0, 1.2*xmax)
        axislegend(ax_4,
            [LineElement(color = :gray, linestyle = :dash, linewidth=1)],
            ["Cook's distance"],
            position = :lb,
            framevisible = false,
            labelsize = 10,
            labelcolor = :gray,
            padding = (0.0f0, 0.0f0, 0.0f0, 0.0f0)
        )
        residualsleverageplot!(ax_4, obj)
        fig
    end

    Makie.@recipe(ResidualPlot, obj) do scene
        Theme(
            axislines = true,
            axislinestyle = :dot,
            axislinecolor = :gray,
            axislinewidth = 1,
        )

    end

    function Makie.plot!(rp::ResidualPlot{<:Tuple{LinearModel}})
        r = residuals(rp.obj[])
        y = predict(rp.obj[])

        scatter!(rp, y, r)
        if rp.axislines[]
            hlines!(rp, [0], color = rp.axislinecolor[], linestyle = rp.axislinestyle[], linewidth = rp.axislinewidth[])
        end
        rp
    end

    Makie.@recipe(ScaleLocationPlot, obj) do scene
        Theme()
    end

    function Makie.plot!(slp::ScaleLocationPlot{<:Tuple{LinearModel}})
        r = standardized_residuals(slp.obj[])
        y = predict(slp.obj[])

        scatter!(slp, y, (sqrt ∘ abs).(r))
        return slp
    end

    Makie.@recipe(ResidualsLeveragePlot, obj) do scene
        Theme(
            axislines = true,
            axislinestyle = :dot,
            axislinecolor = :gray,
            axislinewidth = 1,
            cookslevels = [0.5,1.0],
            cookslinecolor = :gray,
            cookslinestyle = :dash,
            cookslinewidth = 1
        )
    end

    function Makie.plot!(rlp::ResidualsLeveragePlot{<:Tuple{LinearModel}})
        r = standardized_residuals(rlp.obj[])
        h = leverage(rlp.obj[])
        k = dof(rlp.obj[]) - 1

        scatter!(rlp, h, r)

        if rlp.axislines[]
            hlines!(rlp, [0],
                linestyle = rlp.axislinestyle[],
                color = rlp.axislinecolor[],
                linewidth = rlp.axislinewidth[]
            )
            vlines!(rlp, [0],
                linestyle = rlp.axislinestyle[],
                color = rlp.axislinecolor[],
                linewidth = rlp.axislinewidth[]
            )
        end

        if !isempty(rlp.cookslevels[])
            cooksfun = (h,D,k) -> sqrt(D * k * (1-h) / h)
            xmin, xmax = extrema(h)
            xs = LinRange(xmin, 1.1*xmax, 50)
            for D in rlp.cookslevels[]
                lines!(rlp, xs, cooksfun.(xs,D,k),
                    linestyle = rlp.cookslinestyle[],
                    color = rlp.cookslinecolor[],
                    linewidth = rlp.cookslinewidth[]
                )
                text!(rlp, xs[end], cooksfun(xs[end],D,k),
                    text = "$D",
                    color = rlp.cookslinecolor[],
                    #align = (:left, :bottom),
                    offset = (1,-4),
                    fontsize = 10
                )
                lines!(rlp, xs, -cooksfun.(xs,D,k),
                    linestyle = rlp.cookslinestyle[],
                    color = rlp.cookslinecolor[],
                    linewidth = rlp.cookslinewidth[]
                )
                text!(rlp, xs[end], -cooksfun(xs[end],D,k),
                    text = "$D",
                    color = rlp.cookslinecolor[],
                    #align = (:left, :bottom),
                    offset = (1,-4),
                    fontsize = 10
                )
            end
        end

        return rlp
    end

    Makie.@recipe(CooksLeveragePlot, obj) do scene
        Theme()
    end

    function Makie.plot!(clp::CooksLeveragePlot{<:Tuple{LinearModel}})
        h = leverage(clp.obj[])
        cd = cooksdistance(clp.obj[])

        scatter!(clp, h,cd)
        return clp
    end

    function Makie.plot!(qqp::QQPlot{<:Tuple{LinearModel}})
        r = standardized_residuals(qqp[1][])
        qqplot!(qqp, Normal, r,
            qqline = :fitrobust,
            linestyle = :dash,
            linewidth = 1
        )
        return qqp
    end

end
