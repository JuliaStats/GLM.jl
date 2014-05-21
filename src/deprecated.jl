import Base.depwarn

for (fitfn, fittype) in ((:glm, :GeneralizedLinearModel), (:lm, :LinearModel), (:lmc, :(LinearModel{DensePredChol})))
    @eval begin
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

typealias LmMod LinearModel
typealias GlmMod GeneralizedLinearModel
