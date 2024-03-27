@testset "Tests for general functionality of patent model" begin
    @test let
        # Definitely not a test
        using RenewalInference, QuasiMonteCarlo, BenchmarkTools, Plots, InteractiveUtils, Optimization, Distributions, ForwardDiff, OptimizationOptimJL, LineSearches, CSV, DataFrames, KernelDensity, CairoMakie, LinearAlgebra
        using OptimizationBBO, Interpolations, OptimizationNLopt, StatsBase, HypothesisTests
        # ϕ, γ, δ, θ
        # par = [.9, .6, .9, .95];
        # append!(par, [5., .2, .1, 200, 3])
        c = [116, 138, 169, 201, 244, 286, 328, 381, 445, 508, 572, 646, 720, 794, 868, 932, 995];
        # append!(par, [2., 8, 0.1, 0.2, -.3])
        # σ, β11, β12 (μ = β11+ β12x), β21, β22 (σⁱ = β21 + β22x)
        N=30000;
        # X = hcat(ones(N), rand(Normal(μ,σ),N,K))

        par = [.0, .0, .8, 1.];
        append!(par, [2., 8., .1, .2, -.3, 0, 0])
        X=CSV.read("C:/Users/Santeri/Downloads/Deterministic/inv_chars_det_data.csv", DataFrame)
        r_mul = CSV.read("C:/Users/Santeri/Downloads/Deterministic/r_mul.csv", DataFrame)
        # data_stopping = X[:, "renewals_paid"]
        data_stopping = X[:, "renewals"]
        # X=Matrix(X[:,["inventor_age", "sex", "humanities"]])
        X=Matrix(X[:,["age", "sex", "humanities"]])
        # X = Matrix(rand(MvNormal(ones(1),I(1)),30_000)')
        # X = rand(Normal(0, 1), N, 1);
        RenewalInference.initial_shock_parametrisation(par, X)

        dσ = rand(Normal(0, 1), N, 1);

        p0 = [
            Uniform(.5,.8),
            Uniform(.5,.9),
            Uniform(.7,1),
            Uniform(.5,1),
            Uniform(.5,1),
            Uniform(.5,1),
            Uniform(0,5),
            Uniform(0,2_000),
            Uniform(0,10),
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
            alg=Uniform(),
            β=0.
        )
        x=patenthz(par,md_sim)
        md = ModelData(
            x[1],
            Vector{Float64}(c),
            repeat(X,20,1),
            repeat(dσ,20,1),
            controller = ModelControl(),
            β=.0
        )
        # md_sim_dual = ModelData(
        #     zeros(ForwardDiff.Dual, 17),
        #     Vector{ForwardDiff.Dual}(c),
        #     convert(Matrix{ForwardDiff.Dual}, X),
        #     convert(Matrix{ForwardDiff.Dual}, dσ)
        # )
        # ForwardDiff.derivative(a->patenthz([.0, .0, a, 1., 2., 1.8, .1, .2, -.3, 0, 0],md_sim), .5)
        # patenthz([.0, .0, .5, 1., 2., 1.8, .1, .2, -.3, 0, 0],md_sim)
        # ForwardDiff.gradient(a->patenthz([.0, .0, a[1], 1., a[2], a[3], .1, .2, -.3, 0, 0],md), [.5, 1, 5])

        # s = reverse(cumsum(rand(1:50, 10)))
        # s = [284,255,249,246,205,161,114,76,72,24]
        # ForwardDiff.jacobian(s->RenewalInference.modelhz(s, 300), [284,255,249,246,205,161,114,76,0,0])
        # α = 8. #mean of initial returns
        # σ = 2 #standard deviation of initial returns
        # δ = 0.2 #decay rate of returns
        # β = [0.1,0.2,-.3]
        # optF = OptimizationFunction((a,x)->patenthz([.0, .0, a[1], 1., a[2], a[3], .1, .2, -.3, 0, 0],md))
        optF = OptimizationFunction((a,x)->patenthz([.0, .0, .8, 1., 2, 8, .1, .2, a[1], 0, 0],md), Optimization.AutoForwardDiff())
        prob = OptimizationProblem(optF, [.5], [0], )
        @time res = solve(prob, LBFGS())

        ForwardDiff.gradient(a->patenthz(a,md_sim), par)
        ForwardDiff.derivative(a->quantile(LogNormal(a,2), .2), .5)

        x=patenthz(par,md_sim)
        emp_stopping = findfirst.(eachrow(x[end].!=1))
        emp_stopping[emp_stopping.==nothing] .= length(c)+1
        stop_count = [get(countmap(emp_stopping), i, 0) for i in 1:Int(maximum(emp_stopping))]
        Plots.plot(x[1], labels=false)
        
        Plots.bar([get(countmap(emp_stopping), i, 0) for i in 1:Int(maximum(emp_stopping))], alpha=.5, label="Stochastic")
        # Plots.bar(emp_stopping.-[0;emp_stopping[1:end-1]], alpha=.5, label="Stochastic")
        Plots.bar!([get(countmap(data_stopping), i, 0) for i in 1:Int(maximum(data_stopping))], alpha=.5, label="Static")
        
        println(ChisqTest([countmap(emp_stopping)[i] for i in 1:Int(maximum(data_stopping))], [countmap(data_stopping)[i] for i in 1:Int(maximum(data_stopping))]))
        
        # emp_data = hcat(prepare_data(md_sim), emp_stopping)
        # ae_data = AEData(emp_data)
        md = ModelData(
            vec(x[1]),
            Vector{Float64}(c),
            X,
            dσ,
            controller = ModelControl()
        )
        y = patenthz(par,md)
        #     δ    σ  μ   β...
        p0 = [.75, 3, 10, .5,.5,.5]
        optF = OptimizationFunction((a,x)->patenthz([.0, .0, a[1], 1., a[2], a[3], a[4], a[5], a[6], 0, 0],md))
        prob = OptimizationProblem(optF, p0, [0])
        res = solve(prob, NelderMead())


        ### Reduced ###
        opt_patent = OptimizationFunction(
            (a,x)->patenthz([.0, .0, a[1], 1., a[2], a[3], a[4], a[5], a[6], 0, 0],md)
        )
        
        res = Dict()
        @time for i in 1:1 #ϕ=.725:.05:.775, σⁱ=17500:5000:22500, γ=.475:.05:.525, δ=.925:.05:.975, θ=.925:.05:.975
            @show i
            x0 = [.75, 3, 10, .5,.5,.5]
            # x0 = par .+ par./50 .* (-1).^rand(0:1, 9)

            prob = OptimizationProblem(
                opt_patent,
                x0, 
                [0],
                lb = [0.,    0, 0,  0,  0,  0],
                ub = [1.,    Inf, Inf,  1,  1, 1]
            )
            
            @time res[i] = solve(prob, NelderMead())
        end
        ###############

        ### AE ###
        x0 = par .+ par./50 .* (-1).^rand(0:1, 9)
        @time ae_res = Optim.optimize(
            (a)->AEloss(
                a,
                md, 
                ae_data,
                save="C:/Users/Santeri/Desktop/rand-par-ae-estimate.csv",
                save_pred="C:/Users/Santeri/Desktop/rand-par-ae-estimate-pred.csv"
            ),
            [0.,    0, 0,  0,  0,  0,      -Inf,   -Inf,  -Inf],
            [1.,    1, 1,  1,  Inf,  Inf,    Inf,    Inf,    Inf],
            x0, 
            NelderMead(),
            Optim.Options(
                x_tol=1e-2,
                g_tol=1e-2,
                f_tol=1e-2,
                store_trace=true
            )
        )
        ##########
        
        opt_patent = OptimizationFunction(
            (a,x)->patenthz(a,md)
        )
        
        res = Dict()
        @time for i in 1:25 #ϕ=.725:.05:.775, σⁱ=17500:5000:22500, γ=.475:.05:.525, δ=.925:.05:.975, θ=.925:.05:.975
            @show i
            x0 = collect(Iterators.flatten(rand.(p0, 1)))
            # x0 = par .+ par./50 .* (-1).^rand(0:1, 9)

            prob = OptimizationProblem(
                opt_patent,
                x0, 
                [0],
                #     ϕ,  σⁱ,       γ,  δ,  θ,  β₀,     β₁,     β₂,
                lb = [0.,    0, 0,  0,  0,  0,      -Inf,   -Inf,  -Inf],
                ub = [1.,    1, 1,  1,  Inf,Inf,    Inf,    Inf,    Inf]
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
        CSV.write("C:/Users/Santeri/Desktop/29-12-25round.csv", DataFrame(pars', labels))
        data = CSV.read("C:/Users/Santeri/Desktop/9-par-5-round.csv", DataFrame)
        # data = DataFrame(pars', labels)
        RenewalInference.plot_paramdist(data, par)
        save("C:/Users/Santeri/Desktop/param_dist-25-29-12.png", RenewalInference.plot_paramdist(data, par))
        


        Plots.plot(m, color="grey",legend=false)
        Plots.plot!(x[1], color="red", linewidth=2, legend=false)
        savefig("C:/Users/Santeri/Desktop/hz-25-29-12.png")
    end

    @test let
        m=NormModel(50,10,100,collect(35:5:65), .95);
        x=simulate_normhz(m);
        (typeof(x[1])==Vector{Float64})&(typeof(x[2])==Vector{Float64})
    end
end
