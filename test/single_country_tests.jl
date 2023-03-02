using RenewalInference
using Test

@testset "Test single_country!" begin
   @test RenewalInference.test() == "test"
end
