@testset "Tests for general functionality of patent model" begin
    @test let 
        using RenewalInference, QuasiMonteCarlo, BenchmarkTools, Plots, InteractiveUtils, Optimization, Distributions, ForwardDiff, OptimizationOptimJL
        par = [.1, 20000., .1, .95, .95];
        c = [116, 138, 169, 201, 244, 286, 328, 381, 445, 508, 572, 646, 720, 794, 868, 932, 995];
        
        x=_simulate_patenthz(par,c)
        empirical_hz = x[1];

        patent = (a,p)->_patenthz(a, empirical_hz, c)[1]

        p0 = [
            truncated(Normal(.1, .1*.15), 0, 1),
            truncated(Normal(20000, 20000*.15), 0, 100_000),
            truncated(Normal(.1, .1*.15), 0, 1),
            truncated(Normal(.95, .95*.15), 0, 1),
            truncated(Normal(.95, .95*.15), 0, 1)
        ];
        x0 = collect(Iterators.flatten(rand.(p0, 1)))
        patent(x0, 0)[1]

        opt_patent = OptimizationFunction(
            patent,
            Optimization.AutoForwardDiff()
        )

        optp = OptimizationProblem(
            opt_patent,
            collect(Iterators.flatten(rand.(p0, 1))),
            [0],
            lb = [0.,0,0,0,0],
            ub = [1.,100_000,1,1,1]
        )

        i=0
        res = Dict()
        for ϕ=.075:.05:.125, σⁱ=17500:5000:22500, γ=.075:.05:.125, δ=.925:.05:.975, θ=.925:.05:.975
            i+=1
            @show i
            res[i] = optimize(
                a->_patenthz(a, empirical_hz, c, N=750)[1],
                [0.,0,0,0,0],
                [1.,100_000,1,1,1],
                [ϕ, σⁱ, γ, δ, θ]
            )
        end

        succesful_res = [res[i] for i=1:length(res) if res[i].f_converged == 0];

        m = zeros(17,length(succesful_res))
        pars = zeros(5,length(succesful_res))
        for i=eachindex(succesful_res)
            m[:,i] .= _patenthz(succesful_res[i].minimizer, empirical_hz, c)[2]
            pars[:,i] .= succesful_res[i].minimizer
        end
        plot(m)


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