using Distributions, Documenter, GLM, StatsBase

makedocs(; format=Documenter.HTML(),
         sitename="GLM",
         modules=[GLM],
         pages=["Home" => "index.md",
                "examples.md",
                "r-comparison.md",
                "api.md",
                "implementation.md"],
         debug=false,
         doctest=true,
         warnonly=[:missing_docs])

deploydocs(; repo="github.com/JuliaStats/GLM.jl.git",
           push_preview=true)
