@testset "Tests for general functionality of patent model" begin
    @test let 
        par = [.1, 20000., .1, .95, .95, .95, .9, 1000];
        c = [116, 138, 169, 201, 244, 286, 328, 381, 445, 508, 572, 646, 720, 794, 868, 932, 995];
        N=200;T=17;
        simulation_shocks = reshape(QuasiMonteCarlo.sample(N*T,0,1,LowDiscrepancySample(2)),(N,T));
        obsolence = reshape(QuasiMonteCarlo.sample(N*(T-1),0,1,LowDiscrepancySample(2)), (N,T-1));
        ishocks = QuasiMonteCarlo.sample(N,0,1,LowDiscrepancySample(2));
        
        x=simulate_patenthz(
            simulation_shocks, 
            s, c, ishocks, o2 
        );

        fval = patenthz(
            par, x[1],  
        );
        
    end

    @test let 
        m=NormModel(50,10,100,collect(35:5:65), .95);
        x=simulate_normhz(m);
        (typeof(x[1])==Vector{Float64})&(typeof(x[2])==Vector{Float64})
    end
end



"""

Parameter vector order: ϕ, σⁱ, γ, δ, θ, β, ν, N



"""