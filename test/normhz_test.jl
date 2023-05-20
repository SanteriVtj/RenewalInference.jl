using RenewalInference
using Test

@testset "Reuturn object features" begin
    @test let 
        m=RenewalModel(50,10,100,collect(35:5:65));
        x=simulate_normhz(m);
        (length(x[1])==8)&(length(x[2])==100)
    end

    @test let 
        m=RenewalModel(50,10,100,collect(35:5:65));
        x=simulate_normhz(m);
        (typeof(x[1])==Vector{Float64})&(typeof(x[2])==Vector{Float64})
    end
end
