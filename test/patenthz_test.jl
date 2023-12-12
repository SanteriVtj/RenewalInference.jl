@testset "Tests for general functionality of patent model" begin
    @test let
        using RenewalInference, QuasiMonteCarlo, BenchmarkTools, Plots, InteractiveUtils, Optimization, Distributions, ForwardDiff, OptimizationOptimJL, LineSearches, CSV, DataFrames, KernelDensity, CairoMakie, LinearAlgebra
        using OptimizationBBO, Interpolations, OptimizationNLopt
        par = [.9, .4, .95, .95];
        c = [116, 138, 169, 201, 244, 286, 328, 381, 445, 508, 572, 646, 720, 794, 868, 932, 995];
        # append!(par, [2., 8, 0.1, 0.2, -.3])
        append!(par, [2., 2., 1., 20_000, 5])
        N=30000;
        # X = hcat(ones(N), rand(Normal(μ,σ),N,K))

        # X=CSV.read("C:/Users/Santeri/Downloads/Deterministic/inv_chars_det_data.csv", DataFrame)
        # X=Matrix(X[:,["inventor_age", "sex", "humanities"]])
        # X = Matrix(rand(MvNormal(ones(1),I(1)),30_000)')
        X = rand(Normal(10, 1.5), N, 1);

        dσ = rand(Normal(200, 50), N, 1);

        p0 = [
            Uniform(0,1),
            Uniform(0,1),
            Uniform(0,1),
            Uniform(0,1),
            Uniform(0,1),
            Uniform(.001,5),
            Uniform(0,5),
            Uniform(0,100_000),
            Uniform(0,100),
            # Uniform(0,5)
            ];
        x0 = collect(Iterators.flatten(rand.(p0, 1)))

        md_sim = ModelData(
            zeros(Float64, 17),
            Vector{Float64}(c),
            X,
            dσ,
            controller = ModelControl(
                simulation=true
            ),
            alg=Uniform()
        )
        x=patenthz(par,md_sim)
        emp_stopping = sum(x[end], dims=2)
        emp_data = hcat(prepare_data(md_sim), emp_stopping)
        Plots.plot(x[1])

        ae_data = AEData(emp_data)
        md = ModelData(
            vec(x[1]),
            Vector{Float64}(c),
            X,
            dσ,
            controller = ModelControl(
                ae_mode = true
            )
        )
        y = patenthz(par,md)

        x0 = par .+ par./50 .* (-1).^rand(0:1, 9)
        @time ae_res = Optim.optimize(
            (a)->AEloss(
                a,
                md, 
                ae_data,
                save="C:/Users/Santeri/Desktop/rand-par-ae-estimate.csv"
            ),
            [0.,    0, 0,  0,  0,  0,      -Inf,   -Inf,  -Inf],
            [1.,    1, 1,  1,  1,  Inf,    Inf,    Inf,    Inf],
            x0, 
            NelderMead(),
            Optim.Options(
                x_tol=1e-2,
                g_tol=1e-2,
                f_tol=1e-2,
                store_trace=true
            )
        )
        
        opt_patent = OptimizationFunction(
            (a,x)->patenthz(a,md)
        )
        
        res = Dict()
        @time for i in 1:1 #ϕ=.725:.05:.775, σⁱ=17500:5000:22500, γ=.475:.05:.525, δ=.925:.05:.975, θ=.925:.05:.975
            @show i
            # x0 = collect(Iterators.flatten(rand.(p0, 1)))
            x0 = par .+ par./50 .* (-1).^rand(0:1, 9)

            prob = OptimizationProblem(
                opt_patent,
                x0, 
                [0],
                #     ϕ,  σⁱ,       γ,  δ,  θ,  β₀,     β₁,     β₂,
                lb = [0.,    0, 0,  0,  0,  0,      -Inf,   -Inf,  -Inf],
                ub = [1.,    1, 1,  1,  1,  Inf,    Inf,    Inf,    Inf]
            )
            
            @time res[i] = solve(prob, NLopt.LN_NELDERMEAD())
        end
        
        m = zeros(17,length(res));
        pars = zeros(length(res[1].minimizer),length(res));
        loss = zeros(length(res));
        md.controller.debug = true
        for i=eachindex(res)
            m[:,i] .= patenthz(res[i].minimizer, md)[2]
            loss[i] = patenthz(res[i].minimizer, md)[1]
            pars[:,i] .= res[i].minimizer
        end
        labels = ["ϕ","γ","δ","θ","σ","β10","β11","β20","β21"]
        # CSV.write("C:/Users/Santeri/Desktop/9-par-5-round.csv", DataFrame(pars', labels))
        # data = CSV.read("C:/Users/Santeri/Desktop/9-par-5-round.csv", DataFrame)
        data = DataFrame(pars', labels)
        RenewalInference.plot_paramdist(data, par)
        save("C:/Users/Santeri/Desktop/param_dist-5.png", RenewalInference.plot_paramdist(data, par))
        


        Plots.plot(m, color="grey",legend=false)
        Plots.plot!(x[1], color="red", linewidth=2, legend=false)
        savefig("C:/Users/Santeri/Desktop/hz-25.png")
    end

    @test let
        m=NormModel(50,10,100,collect(35:5:65), .95);
        x=simulate_normhz(m);
        (typeof(x[1])==Vector{Float64})&(typeof(x[2])==Vector{Float64})
    end
end
