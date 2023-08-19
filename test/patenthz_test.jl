@testset "Tests for general functionality of patent model" begin
    @test let
        using RenewalInference, QuasiMonteCarlo, BenchmarkTools, Plots, InteractiveUtils, Optimization, Distributions, ForwardDiff, OptimizationOptimJL, LineSearches
        par = [.99, .1, .95, .95];
        c = [116, 138, 169, 201, 244, 286, 328, 381, 445, 508, 572, 646, 720, 794, 868, 932, 995];

        inventor_data = rand(MvNormal([1_200,80],[200. 0; 0 15]), 200)'

        inv_coef = [10., 75]
        par = vcat(par, inv_coef)
        x=RenewalInference._simulate_patenthz(par,c, inventor_data)
        empirical_hz = x[1];

        patent = (a,p)->RenewalInference._patenthz(a, empirical_hz, c, inventor_data)[1]

        p0 = [
            truncated(Normal(.99, .1*.15), 0, 1),
            truncated(Normal(20000, 20000*.15), 0, 100_000),
            truncated(Normal(.1, .1*.15), 0, 1),
            truncated(Normal(.95, .95*.15), 0, 1),
            truncated(Normal(.95, .95*.15), 0, 1)
        ];
        x0 = collect(Iterators.flatten(rand.(p0, 1)))
        patent(vcat(x0, inv_coef), 0)[1]

        opt_patent = OptimizationFunction(
            patent,
            Optimization.AutoForwardDiff()
        )
        # opt_patent = OptimizationFunction(
        #     patent
        # )

        optp = OptimizationProblem(
            opt_patent,
            [0.075, 0.075, .925, .925, ],
            [0],
            lb = [0.,0,0,0,0],
            ub = [1.,100_000,1,1,1]
        )

        # res = solve(optp, LBFGS(linesearch=LineSearches.BackTracking()))
        res = solve(optp, NelderMead())

        res = optimize(
            a->RenewalInference._patenthz(a, empirical_hz, c)[1],
            [0.,0,0,0,0],
            [1.,100_000,1,1,1],
            x0
        )

        i=0
        res = Dict()
        for ϕ=.075:.05:.125, σⁱ=17500:5000:22500, γ=.075:.05:.125, δ=.925:.05:.975, θ=.925:.05:.975
            i+=1
            @show i
            # res[i] = optimize(
            #     a->RenewalInference._patenthz(a, empirical_hz, c, N=750)[1],
            #     [0.,0,0,0,0],
            #     [1.,100_000,1,1,1],
            #     [ϕ, σⁱ, γ, δ, θ],
            #     method = LBFGS(linesearch = LineSearches.BackTracking()),
            #     autodiff = :forward
            # )

            optp = OptimizationProblem(
                opt_patent,
                [ϕ, σⁱ, γ, δ, θ],
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

        succesful_res = [res[i] for i=1:length(res) if res[i].f_converged == 0];
        # succesful_res = [res[i] for i=1:length(res) if res[i].original.f_converged == 0];

        m = zeros(17,length(succesful_res));
        pars = zeros(5,length(succesful_res));
        for i=eachindex(succesful_res)
            m[:,i] .= RenewalInference._patenthz(succesful_res[i].original.minimizer, empirical_hz, c)[2]
            pars[:,i] .= succesful_res[i].original.minimizer
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