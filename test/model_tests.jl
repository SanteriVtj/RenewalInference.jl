using RenewalInference
using Test

@testset "Model object" begin
    @test let 
        typeof(NormModel(50,10,100,collect(35:5:65), .95))<:NormModel
    end
end
