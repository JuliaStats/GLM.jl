using Distributions, Documenter, GLM, StatsBase

makedocs(
    format = Documenter.HTML(),
    sitename = "GLM",
    modules = [GLM]
)

deploydocs(
    repo   = "github.com/JuliaStats/GLM.jl.git",
)