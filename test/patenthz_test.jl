@testset "Tests for general functionality of patent model" begin
    @test let
        using RenewalInference, QuasiMonteCarlo, BenchmarkTools, Plots, InteractiveUtils, Optimization, Distributions, ForwardDiff, OptimizationOptimJL, LineSearches, CSV, DataFrames, KernelDensity, CairoMakie, LinearAlgebra
        using OptimizationBBO, Interpolations, OptimizationNLopt
        par = [.2, 10000., .3, .9, .95];
        c = [116, 138, 169, 201, 244, 286, 328, 381, 445, 508, 572, 646, 720, 794, 868, 932, 995];
        # append!(par, [2., 8, 0.1, 0.2, -.3])
        append!(par, [2., 5, 0])
        N=30000;
        # X = hcat(ones(N), rand(Normal(μ,σ),N,K))

        # X=CSV.read("C:/Users/Santeri/Downloads/Deterministic/inv_chars_det_data.csv", DataFrame)
        # X=Matrix(X[:,["inventor_age", "sex", "humanities"]])
        # X = Matrix(rand(MvNormal(ones(1),I(1)),30_000)')
        X = Matrix(zeros(N,1));

        dσ = zeros(N,3);
        for (i,v) in enumerate(rand(1:3,N))
            dσ[i,v] = 1
        end

        p0 = [
            Uniform(0,1),
            Uniform(0,100_000),
            Uniform(0,1),
            Uniform(0,1),
            Uniform(0,1),
            Uniform(.001,5),
            Uniform(0,100),
            Uniform(0,15),
            # Uniform(0,5),
            # Uniform(0,5)
            ];
        x0 = collect(Iterators.flatten(rand.(p0, 1)))

        md_sim = ModelData(
            zeros(Float64, 17),
            Vector{Float64}(c),
            X,
            controller = ModelControl(
                true,   # simulation
                false,  # x_transformed
                true    # debug
            ),
            alg=Uniform()
        )
        x=patenthz(par,md_sim)
        Plots.plot(x[1])

        # md = ModelData(
        #     vec(x[1]),
        #     Vector{Float64}(c),
        #     X
        # )
        md = ModelData(
            collect(LinRange(.65, .2, 17)),
            Vector{Float64}(c),
            X
        )
        y = patenthz(par,md)
        
        opt_patent = OptimizationFunction(
            (a,x)->patenthz(a,md),
            Optimization.AutoForwardDiff(),
        )
        
        res = Dict()
        @time for i in 1:25 #ϕ=.725:.05:.775, σⁱ=17500:5000:22500, γ=.475:.05:.525, δ=.925:.05:.975, θ=.925:.05:.975
            @show i
            x0 = collect(Iterators.flatten(rand.(p0, 1)))
            # x0 = par .+ par./20 .* (-1).^rand(0:1, 10)

            # prob = OptimizationProblem(
            #     opt_patent,
            #     x0, 
            #     [0],
            #     #     ϕ,  σⁱ,       γ,  δ,  θ,  β₀,     β₁,     β₂,     β₃,     β₄
            #     lb = [0.,    0,          0,  0,  0,  0,      -Inf,   -Inf,   -Inf,   -Inf],
            #     ub = [1.,    100_000,    1,  1,  1,  Inf,    Inf,    Inf,    Inf,    Inf]
            # )
            prob = OptimizationProblem(
                opt_patent,
                x0, 
                [0],
                #     ϕ,  σⁱ,       γ,  δ,  θ,  β₀,     β₁,     β₂,
                lb = [0.,    0,          0,  0,  0,  0,      -Inf,   -Inf],
                ub = [1.,    100_000,    1,  1,  1,  Inf,    Inf,    Inf]
            )
            @time res[i] = solve(prob, NLopt.LN_NELDERMEAD(), reltol=1e-6)
        end
        
        m = zeros(17,length(res));
        pars = zeros(10,length(res));
        loss = zeros(length(res));
        md.controller.debug = true
        for i=eachindex(res)
            m[:,i] .= patenthz(res[i].minimizer, md)[2]
            loss[i] = patenthz(res[i].minimizer, md)[1]
            pars[:,i] .= res[i].minimizer
        end
        labels = ["ϕ","σⁱ","γ","δ","θ","σ","β0","β1","β2","β3"]
        CSV.write("C:/Users/Santeri/Desktop/10-par-25-round.csv", DataFrame(pars', labels))
        data = CSV.read("C:/Users/Santeri/Desktop/10-par-25-round.csv", DataFrame)
        RenewalInference.plot_paramdist(data, par)
        save("C:/Users/Santeri/Desktop/param_dist-25.png", RenewalInference.plot_paramdist(data, par))
        


        Plots.plot(m, color="grey",legend=false)
        Plots.plot!(x[1], color="red", linewidth=2, legend=false)
        savefig("C:/Users/Santeri/Desktop/hz-25.png")
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

Plots.plot(
    [0,1],
    [.75,.4]
)
Plots.plot!(
    [0,1],
    [.65,.55],
    xtickfontcolor=:white,
    ytickfontcolor=:white,
    xlim=(0,1),
    ylim=(0,1),
    legend=false,
    xlabel=L"\varepsilon",
    ylabel=L"EU"
)
annotate!(-.045, .75, L"u_a")
annotate!(1-.045, .4, L"u_c")
annotate!(-.045, .65, L"u_b")
annotate!(1-.045, .55, L"u_d")
savefig("C:/Users/Santeri/Desktop/Behvioral-theory-hw3.png")
