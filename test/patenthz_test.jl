@testset "Tests for general functionality of patent model" begin
    @test let
        using RenewalInference, QuasiMonteCarlo, BenchmarkTools, Plots, InteractiveUtils, Optimization, Distributions, ForwardDiff, OptimizationOptimJL, LineSearches, CSV, DataFrames, KernelDensity, CairoMakie
        par = [.2, 20000., .3, .9, .95];
        c = [116, 138, 169, 201, 244, 286, 328, 381, 445, 508, 572, 646, 720, 794, 868, 932, 995];
        append!(par, [2., 8, 0.1, 0.2, -.3])
        μ=1;σ=5;N=30000;K=3
        # X = hcat(ones(N), rand(Normal(μ,σ),N,K))

        X=CSV.read("C:/Users/Santeri/Downloads/Deterministic/inv_chars_det_data.csv", DataFrame)
        X=Matrix(X[:,["inventor_age", "sex", "humanities"]])
        # X = hcat(ones(1000), rand(Normal(5,1),1000))

        p0 = [
            Uniform(0,1),
            Uniform(0,100_000),
            Uniform(0,1),
            Uniform(0,1),
            Uniform(0,1),
            Uniform(.001,5),
            Uniform(0,10),
            Uniform(0,5),
            Uniform(0,5),
            Uniform(0,5)
            ];
        x0 = collect(Iterators.flatten(rand.(p0, 1)))

        md = ModelData(
            zeros(Float64, 17),
            Vector{Float64}(c),
            X,
            controller = ModelControl(
                true,   # simulation
                false,  # x_transformed
                true    # debug
            )
        )
        x=patenthz(par,md)
        md.controller.simulation = false
        y = patenthz(par,md)
        md.controller.debug = false
        
        opt_patent = OptimizationFunction(
            a->patenthz(a,md),
            Optimization.AutoForwardDiff(),
        )
        
        res = Dict()
        @time for i in 1:1 #ϕ=.725:.05:.775, σⁱ=17500:5000:22500, γ=.475:.05:.525, δ=.925:.05:.975, θ=.925:.05:.975
            @show i
            x0 = collect(Iterators.flatten(rand.(p0, 1)))
            x0 = par .+ par./100 .* (-1).^rand(0:1, 10)

            prob = OptimizationProblem(
                patent,
                x0, 
                [0],
                #     ϕ,  σⁱ,       γ,  δ,  θ,  β₀,     β₁,     β₂,     β₃,     β₄
                lb = [0.,    0,          0,  0,  0,  0,      -Inf,   -Inf,   -Inf,   -Inf],
                ub = [1.,    100_000,    1,  1,  1,  Inf,    Inf,    Inf,    Inf,    Inf]
            )
            res[i] = solve(prob, LBFGS())

            # res[i] = optimize(
            #     patent,
            #     #     ϕ,  σⁱ,       γ,  δ,  θ,  β₀,     β₁,     β₂,     β₃,     β₄
            #     [0.,    0,          0,  0,  0,  0,      -Inf,   -Inf,   -Inf,   -Inf],
            #     [1.,    100_000,    1,  1,  1,  Inf,    Inf,    Inf,    Inf,    Inf],
            #     x0,
            #     Fminbox(NelderMead()),
            #     Optim.Options(
            #         f_abstol=1e-4,
            #         x_abstol=1e-4
            #     )
            # )
            # time_limit=60
        end

        succesful_res = [res[i] for i=1:length(res) if res[i].f_converged == 0];
        # succesful_res = [
        #     res[i] 
        #     for i=1:length(res) 
        #     if (res[i].original.f_converged == 0)&all(res[i].original.minimizer .≠ res[i].original.initial_x)];
        
        
        m = zeros(17,length(succesful_res));
        pars = zeros(8,length(succesful_res));
        for i=eachindex(succesful_res)
            m[:,i] .= RenewalInference._patenthz(succesful_res[i].original.minimizer, empirical_hz, c)[2]
            pars[:,i] .= succesful_res[i].original.minimizer
        end
        labels = ["ϕ","σⁱ","γ","δ","θ","σ","β0","β1"]
        CSV.write("C:/Users/Santeri/Desktop/pakes-74.csv", DataFrame(pars', labels))
        data = CSV.read("C:/Users/Santeri/Desktop/pakes-74.csv", DataFrame)
        RenewalInference.plot_paramdist(data, [.75, 20000,.5,.95,.95,1,0,.5])
        


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