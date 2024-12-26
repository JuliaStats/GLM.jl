module MakieExt
    using Makie
    using GLM
    using Distributions
    using GLM: leverage, standardized_residuals
    import GLM: cooksleverageplot, cooksleverageplot!
    import GLM: scalelocationplot, scalelocationplot!
    import GLM: residualplot, residualplot!
    import GLM: residualsleverageplot, residualsleverageplot!
    import GLM: quantilequantileplot, quantilequantileplot!

    Makie.@recipe(ResidualPlot, obj) do scene
        Theme(
            axislines = true,
            axislinestyle = :dot,
            axislinecolor = :gray,
            axislinewidth = 1,
            axislabels = true,
            axistitle = true
        )
        #Attributes()
    end

    function Makie.plot!(rp::ResidualPlot{<:Tuple{LinearModel}})
        r = residuals(rp.obj[])
        y = predict(rp.obj[])

        ax = current_axis()

        if rp.axistitle[]
            ax.title = "Residuals vs Fitted"
        end
        if rp.axislabels[]
            ax.xlabel = "Fitted Values"
            ax.ylabel = "Residuals"
        end

        scatter!(rp, y, r)
        if rp.axislines[]
            hlines!(rp, [0], color = rp.axislinecolor[], linestyle = rp.axislinestyle[], linewidth = rp.axislinewidth[])
        end
        rp
    end

    Makie.@recipe(ScaleLocationPlot, obj) do scene
        Theme(
            axislabels = true,
            axistitle = true
        )
    end

    function Makie.plot!(slp::ScaleLocationPlot{<:Tuple{LinearModel}})
        ax = current_axis()
        if slp.axistitle[]
            ax.title = "Scale-Location"
        end
        if slp.axislabels[]
            ax.xlabel = "Fitted Values"
            ax.ylabel = "√|standardized residuals|"
        end

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
            axislabels = true,
            axistitle = true,
            cookslevels = [0.5,1.0],
            cookslinecolor = :gray,
            cookslinestyle = :dash,
            cookslinewidth = 1
        )
    end

    function Makie.plot!(rlp::ResidualsLeveragePlot{<:Tuple{LinearModel}})
        ax = current_axis()

        if rlp.axistitle[]
            ax.title = "Residuals vs Leverage"
        end
        if rlp.axislabels[]
            ax.xlabel = "Leverage"
            ax.ylabel = "Standardized Residuals"
        end

        r = standardized_residuals(rlp.obj[])
        h = leverage(rlp.obj[])
        k = dof(rlp.obj[]) - 1

        ymax = maximum(abs.(r))
        ylims!(ax, -1.1*ymax, 1.1*ymax)

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
                lines!(rlp, xs, -cooksfun.(xs,D,k),
                    linestyle = rlp.cookslinestyle[],
                    color = rlp.cookslinecolor[],
                    linewidth = rlp.cookslinewidth[]
                )
            end
        end

        scatter!(rlp, h, r)
        return rlp
    end

    Makie.@recipe(CooksLeveragePlot, obj) do scene
        Theme(
            axistitle = true,
            axislabels = true,
        )
    end

    function Makie.plot!(clp::CooksLeveragePlot{<:Tuple{LinearModel}})
        ax = current_axis()
        if clp.axistitle[]
            ax.title = "Cook's distance vs Leverage"
        end
        if clp.axislabels[]
            ax.xlabel = "Leverage"
            ax.ylabel = "Cook's Distance"
        end

        h = leverage(clp.obj[])
        cd = cooksdistance(clp.obj[])

        scatter!(clp, h,cd)
        return clp
    end

    function Makie.plot!(qqp::QQPlot{<:Tuple{LinearModel}})
        ax = current_axis()

        ax.ylabel = "Standardized Residuals"
        ax.xlabel = "Theoretical Quantiles"
        ax.title = "Q-Q Residuals"

        r = standardized_residuals(qqp[1][])
        qqplot!(qqp, Normal, r,
            qqline = :identity,
            linestyle = :dash,
            linewidth = 1
        )
        return qqp
    end
end
