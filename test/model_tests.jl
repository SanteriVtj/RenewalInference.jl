using RenewalInference
using Test

@testset "Model object" begin
    @test let 
        model = RenewalModel{Int64}(
            rand(1:20, 1500),
            rand(20),
            20,
            100,
            rand(5)
        );
        typeof(model) == RenewalModel{Int64}
    end

    @test_throws ErrorException("Fees need to be defined for each period") RenewalModel{Int64}(
        rand(1:20, 1500),
        rand(21),
        20,
        100,
        rand(5)
    )
end


