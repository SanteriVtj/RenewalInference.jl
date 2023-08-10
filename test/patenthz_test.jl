@testset "Tests for general functionality of patent model" begin
    @test let 
        # using RenewalInference, QuasiMonteCarlo, BenchmarkTools, Plots, InteractiveUtils, Optimization, Distributions, Optim
        par = [.1, 20000., .1, .95, .95];
        c = [116, 138, 169, 201, 244, 286, 328, 381, 445, 508, 572, 646, 720, 794, 868, 932, 995];
        N=200;T=17;
        simulation_shocks = QuasiMonteCarlo.sample(N,T,QuasiMonteCarlo.HaltonSample())';
        obsolence = QuasiMonteCarlo.sample(N,T-1,QuasiMonteCarlo.HaltonSample())';
        ishock = QuasiMonteCarlo.sample(N,1,QuasiMonteCarlo.HaltonSample())';
        
        x=simulate_patenthz(
            par,
            simulation_shocks,
            obsolence,
            c,
            ishock
        );

        empirical_hz = x[1];

        opt_f = (a,p)->patenthz(
            a,
            empirical_hz,
            ishock,
            obsolence,
            c
        )[1]

        p0 = [
            truncated(Normal(.1, .1*.05), 0, 1),
            truncated(Normal(20000, 20000*.05), 0, 100_000),
            truncated(Normal(.1, .1*.05), 0, 1),
            truncated(Normal(.95, .95*.05), 0, 1),
            truncated(Normal(.95, .95*.05), 0, 1)
        ];

        # res=optimize(
        #     init0->patenthz(
        #         init0,
        #         empirical_hz,
        #         ishock,
        #         obsolence,
        #         c
        #     )[1],
        #     collect(Iterators.flatten(rand.(p0, 1)))
        # )

        optp = OptimizationProblem(
            opt_f,
            collect(Iterators.flatten(rand.(p0, 1))),
            [0],
            lb = [0.,0,0,0,0],
            ub = [1.,100_000,1,1,1]
        );

        y = solve(
            optp,
            BFGS()
        )

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