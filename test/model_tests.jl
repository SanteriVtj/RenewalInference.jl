using RenewalInference
using Test

@testset "Model object" begin
    @test let 
        m=NormModel(50,10,100,collect(35:5:65), .95);
        typeof(m) == NormModel
    end
end
