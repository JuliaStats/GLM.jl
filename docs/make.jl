using Distributions, Documenter, GLM, StatsBase

makedocs(
    format = Documenter.HTML(),
    sitename = "GLM",
    modules = [GLM],
    pages = [
        "Home" => "index.md",
        "manual.md",
        "examples.md",
        "api.md",
    ],
    debug = false,
)

deploydocs(
    repo   = "github.com/JuliaStats/GLM.jl.git",
)