using RenewalInference
using Test

@testset "Model object" begin
    @test let 
        m=RenewalModel(50,10,100,collect(35:5:65));
        typeof(m) == RenewalModel
    end
end
