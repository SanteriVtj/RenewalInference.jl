@testset "Tests for general functionality of patent model" begin
    @test let
        # Definitely not a test
        using Revise, RenewalInference, QuasiMonteCarlo, BenchmarkTools, Plots, InteractiveUtils, Optimization, Distributions, ForwardDiff, OptimizationOptimJL, CSV, DataFrames, KernelDensity, CairoMakie, LinearAlgebra
        using Interpolations, StatsBase, HypothesisTests, LaTeXStrings, Measures, Debugger, StructArrays, ProfileView, Random, Optim
        Random.seed!(1729)
        
        c = [116, 138, 169, 201, 244, 286, 328, 381, 445, 508, 572, 646, 720, 794, 868, 932, 995];
        # σ, β11, β12 (μ = β11+ β12x), β21, β22 (σⁱ = β21 + β22x)
        N=1000;T=length(c)
        
        par = [.85, 10., .85, .95, 2, 2, .7, 3, 2, 50, 150]
        X=CSV.read("C:/Users/Santeri/Downloads/Deterministic/inv_chars_det_data.csv", DataFrame)
        data_stopping = X[1:N, "renewals"]
        data_stopping = min.(T,max.(data_stopping,1))
        X = Matrix(X[1:N,["age", "sex","humanities"]])
        X = hcat(ones(N),X)
        r_mul = CSV.read("C:/Users/Santeri/Downloads/Deterministic/r_mul.csv", DataFrame)
        # data_stopping = X[:, "renewals_paid"]
        # X=Matrix(X[:,["inventor_age", "sex", "humanities"]])
        # X=zeros(N,3)
        # X=Matrix(X[1:N,["age", "sex", "humanities"]])
        # X = zeros(N,1)
        # X = Matrix(rand(MvNormal(ones(1),I(1)),30_000)')
        # X = rand(Normal(0, 1), N, 1);

        dσ = float.(rand(Bernoulli(.75), N, 1));
        dσ = hcat(ones(N),dσ)

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
            zeros(N),
            alg=Uniform()
        )
        r, r_d = gen_sample(par,md_sim)
        # CSV.write("simulation/r-24-8.csv",DataFrame(r,:auto))
        # CSV.write("simulation/r_d-24-8.csv",DataFrame(r_d,:auto))
        # open("simulation/par-24-8.txt","w") do f
        #     write(f, join(par," "))
        # end
        Plots.histogram(sum(r_d,dims=2),bins=17,legends=false)
        Plots.plot(RenewalInference.modelhz(sum(r_d,dims=1)',N),ylims=(0,1))
        mean(r,dims=1)

        renewals = findfirst.(eachrow(r_d.==0))
        renewals[isnothing.(renewals)] .= T
        renewals = convert(Vector{Float64}, renewals)
        md = ModelData(
            zeros(Float64, 17),
            Vector{Float64}(c),
            X,
            dσ,
            renewals
        )
        fval(par,md,S=100)
        res =  optimize(
            a->fval([a[1], 10., .85, .95, 2, 2, .7, 3, 2, 10, 10], md, S=250),
            rand(1),
            NelderMead()
        )
        
        optimize_n(a->a^2,[Uniform()],10)
        


        
       
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
