using RenewalInference
using Test

@testset "Model object" begin
    @test let 
        model = RenewalModel(
            1,
            .5
        );
        typeof(model) == RenewalModel
    end
end
