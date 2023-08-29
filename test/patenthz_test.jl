@testset "Tests for general functionality of patent model" begin
    @test let
        using RenewalInference, QuasiMonteCarlo, BenchmarkTools, Plots, InteractiveUtils, Optimization, Distributions, ForwardDiff, OptimizationOptimJL, LineSearches, CSV, DataFrames, KernelDensity, CairoMakie
        par = [.75, 20000., .5, .95, .95];
        c = [116, 138, 169, 201, 244, 286, 328, 381, 445, 508, 572, 646, 720, 794, 868, 932, 995];

        x=RenewalInference._simulate_patenthz(par,c,N=1000)
        empirical_hz = x[1];

        patent = (a,p)->RenewalInference._patenthz(a, empirical_hz, c, N=1000)[1]

        # p0 = [
        #     truncated(Normal(.75, 1*.15), 0, 1),
        #     truncated(Normal(20000, 100_000*.15), 0, 100_000),
        #     truncated(Normal(.5, 1*.15), 0, 1),
        #     truncated(Normal(.95, 1*.15), 0, 1),
        #     truncated(Normal(.95, 1*.15), 0, 1)
        # ];
        p0 = [
            Uniform(0,1),
            Uniform(0,100_000),
            Uniform(0,1),
            Uniform(0,1),
            Uniform(0,1),
        ];
        x0 = collect(Iterators.flatten(rand.(p0, 1)))
        patent(x0, 0)[1]

        opt_patent = OptimizationFunction(
            patent,
            Optimization.AutoForwardDiff()
        )

        optp = OptimizationProblem(
            opt_patent,
            x0, # [0.075, 22500, 0.075, .925, .925],
            [0],
            lb = [0.,0,0,0,0],
            ub = [1.,100_000,1,1,1]
        )

        res = solve(optp, LBFGS(linesearch=LineSearches.BackTracking()))
        # res = solve(optp)

        # res = optimize(
        #     a->RenewalInference._patenthz(a, empirical_hz, c)[1],
        #     [0.,0,0,0,0],
        #     [1.,100_000,1,1,1],
        #     x0
        # )

        # i=0
        res = Dict()
        @time for i in 1:150 #ϕ=.725:.05:.775, σⁱ=17500:5000:22500, γ=.475:.05:.525, δ=.925:.05:.975, θ=.925:.05:.975
            # i+=1
            @show i
            # res[i] = optimize(
            #     a->RenewalInference._patenthz(a, empirical_hz, c, N=750)[1],
            #     [0.,0,0,0,0],
            #     [1.,100_000,1,1,1],
            #     [ϕ, σⁱ, γ, δ, θ],
            #     method = LBFGS(linesearch = LineSearches.BackTracking()),
            #     autodiff = :forward
            # )
            x0 = collect(Iterators.flatten(rand.(p0, 1)))
            optp = OptimizationProblem(
                opt_patent,
                x0,# [ϕ, σⁱ, γ, δ, θ],
                [0],
                lb = [0.,0,0,0,0],
                ub = [1.,100_000,1,1,1]
            )

            res[i] = solve(
                optp,
                LBFGS(linesearch=LineSearches.BackTracking()),
                maxtime = 60
            )
        end

        # succesful_res = [res[i] for i=1:length(res) if res[i].f_converged == 0];
        succesful_res = [
            res[i] 
            for i=1:length(res) 
            if (res[i].original.f_converged == 0)&all(res[i].original.minimizer .≠ res[i].original.initial_x)];
        
        
        m = zeros(17,length(succesful_res));
        pars = zeros(5,length(succesful_res));
        for i=eachindex(succesful_res)
            m[:,i] .= RenewalInference._patenthz(succesful_res[i].original.minimizer, empirical_hz, c)[2]
            pars[:,i] .= succesful_res[i].original.minimizer
        end
        labels = ["ϕ","σⁱ","γ","δ","θ"]
        CSV.write("C:/Users/Santeri/Desktop/lbfgs-rng-15-of-interval.csv", DataFrame(pars', labels))
        data = CSV.read("C:/Users/Santeri/Desktop/lbfgs-unif-150.csv", DataFrame)
        RenewalInference.plot_paramdist(data, [.75, 20000,.5,.95,.95])
        


        plot(m)
        labels = ["ϕ","σⁱ","γ","δ","θ"]
        CSV.write("C:/Users/Santeri/Desktop/lbfgs-75-20-50-95-95.csv", DataFrame(pars', labels))
        kde(par'[:,1])

        data = CSV.read("C:/Users/Santeri/Desktop/lbfgs-75-20-50-95-95.csv", DataFrame)
        data = rename(data, Dict(zip(names(data), labels)))
        RenewalInference.plot_paramdist(data, [.75, 20000,.5,.95,.95])


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