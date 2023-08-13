@testset "Tests for general functionality of patent model" begin
    @test let 
        using RenewalInference, BenchmarkTools
        par = [.1, 20000., .1, .95, .95];
        c = [116, 138, 169, 201, 244, 286, 328, 381, 445, 508, 572, 646, 720, 794, 868, 932, 995];
        @benchmark RenewalInference._thresholds(par,c)
    end
end