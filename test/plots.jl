using Test
using GLM
using StatsPlots
using CairoMakie
#using Makie
using GLM: standardized_residuals, leverage
import GLM.PlotsRecipes
import GLM.MakieRecipes


@testset "Utility functions" begin
    rng = StableRNG(2025)
    X = randn(rng,10,3)
    y = randn(rng,10)
    l = lm(X,y)
    h = leverage(l)
    r = standardized_residuals(l)
    @test all(!isnan, r)
    @test all(>=(0.0), h)
end

@testset "StatsPlots Recipes" begin
    # NB. These tests follow the tests of StatsPlots. They mostly check that the functions don't crash
    rng = StableRNG(2025)
    X = randn(rng,10,3)
    y = randn(rng,10)
    l = lm(X,y)
    @testset "residualplot" begin
        pl = PlotsRecipes.residualplot(l)
        @test show(devnull, pl) isa Nothing
        PlotsRecipes.residualplot!(pl, l)
        @test show(devnull, pl) isa Nothing
    end
    @testset "residualsleverageplot" begin
        pl = PlotsRecipes.residualsleverageplot(l)
        @test show(devnull, pl) isa Nothing
        PlotsRecipes.residualsleverageplot!(pl, l)
        @test show(devnull, pl) isa Nothing
    end
    @testset "scalelocationplot" begin
        pl = PlotsRecipes.scalelocationplot(l)
        @test show(devnull, pl) isa Nothing
        PlotsRecipes.scalelocationplot!(pl, l)
        @test show(devnull, pl) isa Nothing
    end
    @testset "qqplot" begin
        pl = StatsPlots.qqplot(l)
        @test show(devnull, pl) isa Nothing
        StatsPlots.qqplot!(pl, l)
        @test show(devnull, pl) isa Nothing
    end
    @testset "qqnorm" begin
        pl = StatsPlots.qqnorm(l)
        @test show(devnull, pl) isa Nothing
        StatsPlots.qqnorm!(pl, l)
        @test show(devnull, pl) isa Nothing
    end

    @testset "cooksleverageplot" begin
        pl = PlotsRecipes.cooksleverageplot(l)
        @test show(devnull, pl) isa Nothing
        PlotsRecipes.cooksleverageplot!(pl, l)
        @test show(devnull, pl) isa Nothing
    end
    @testset "lmplot" begin
        pl = PlotsRecipes.lmplot(l)
        @test show(devnull, pl) isa Nothing
    end
end

@testset "Makie Recipes" begin
    rng = StableRNG(2025)
    X = randn(rng, 10, 3)
    y = randn(rng, 10)
    l = lm(X,y)
    @testset "residualplot" begin
        fig, ax, plt = MakieRecipes.residualplot(l)
        @test plt isa Makie.Plot
    end
    @testset "residualsleverageplot" begin
        fig, ax, plt = MakieRecipes.residualsleverageplot(l)
        @test plt isa Makie.Plot
    end
    @testset "scalelocationplot" begin
        fig, ax, plt = MakieRecipes.scalelocationplot(l)
        @test plt isa Makie.Plot
    end
    @testset "qqplot" begin
        fig, ax, plt = CairoMakie.qqplot(l)
        @test plt isa Makie.Plot
    end
    @testset "cooksleverageplot" begin
        fig, ax, plt = MakieRecipes.cooksleverageplot(l)
        @test plt isa Makie.Plot
    end
    @testset "lmplot" begin
        fig = MakieRecipes.lmplot(l)
        @test fig isa Makie.Figure

    end


end
