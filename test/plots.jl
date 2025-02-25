using Test
using GLM
using StatsPlots
using GLM: standardized_residuals, leverage


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
        pl = residualplot(l)
        @test show(devnull, pl) isa Nothing
    end
    @testset "residualsleverageplot" begin
        pl = residualsleverageplot(l)
        @test show(devnull, pl) isa Nothing
    end
    @testset "scalelocationplot" begin
        pl = scalelocationplot(l)
        @test show(devnull, pl) isa Nothing
    end
    @testset "qqplot" begin
        pl = qqplot(l)
        @test show(devnull, pl) isa Nothing
    end
    @testset "cooksleverageplot" begin
        pl = cooksleverageplot(l)
        @test show(devnull, pl) isa Nothing
    end
    @testset "lmplot" begin
        pl = lmplot(l)
        @test show(devnull, pl) isa Nothing
    end
end

@testset "Makie Recipes" begin

end
