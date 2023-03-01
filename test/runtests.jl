using RenewalInference
using Test

@testset "Model object" begin
    @test let 
        model = RenewalModel{Int64}(
            rand(1:20, 1500),
            20,
            100,
            rand(5)
        );
        typeof(model) == RenewalModel{Int64}
    end
end
