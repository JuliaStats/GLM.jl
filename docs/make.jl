using Distributions, Documenter, GLM

makedocs(
    format = :html,
    sitename = "GLM"
)

deploydocs(
    repo   = "github.com/JuliaStats/GLM.jl.git",
    julia  = "0.6",
    target = "build",
    deps   = nothing,
    make   = nothing
)