import Base.depwarn

for (fitfn, fittype) in ((:glm, :GlmMod), (:lm, :LmMod), (:lmc, :(LmMod{DensePredChol})))
    @eval begin
        function $fitfn(f, df, args...; kwargs...)
            depwarn($("$(string(fitfn))(f::Formula, df::AbstractDataFrame, ...) is deprecated, use fit($(string(fittype)), f::Formula, df::AbstractDataFrame, ...) instead"), $(Base.Meta.quot(fitfn)))
            fit($fittype, f, df, args...; kwargs...)
        end

        function $fitfn(e::Expr, df, args...; kwargs...)
            depwarn($("$(string(fitfn))(e::Expr, df::AbstractDataFrame, ...) is deprecated, use fit($(string(fittype)), f::Formula, df::AbstractDataFrame, ...) instead"), $(Base.Meta.quot(fitfn)))
            eval(quote; import DataFrames; end)
            fit($fittype, DataFrames.Formula(e), df, args...; kwargs...)
        end

        function $fitfn(s::String, df, args...; kwargs...)
            depwarn($("$(string(fitfn))(e::String, df::AbstractDataFrame, ...) is deprecated, use fit($(string(fittype)), f::Formula, df::AbstractDataFrame, ...) instead"), $(Base.Meta.quot(fitfn)))
            eval(quote; import DataFrames; end)
            fit($fittype, DataFrames.Formula(parse(s)[1]), df, args...; kwargs...)
        end
    end
end
