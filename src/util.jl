function plot_paramdist(x, real; cols = 2)
    names = DataFrames.names(x)

    rows = mod(length(names), 2) == 0 ? length(names) ÷ cols : length(names) ÷ 2+1

    fig = Figure(size = (400*rows, 600*cols))
    axs = [Axis(fig[i,j]) for i=1:rows, j=1:cols]

    res = Dict(i=>kde(x[:,i]) for i in names)

    means = mean.(eachcol(x))

    relative_error = abs.(mean.(eachcol((x .- real')./real')))

    for i in eachindex(names)
        lines!(
            axs[i],
            res[names[i]].x,
            res[names[i]].density
        )
        scatter!(
            axs[i],
            x[:,names[i]],
            repeat([0], size(x, 1))
        )
        vlines!(
            axs[i],
            means[i],
            color = :red,
            label = L"\hat{\mu}"
        )
        vlines!(
            axs[i],
            real[i],
            color = :green,
            label = L"\mu*"
        )
        text!(
            axs[i],
            means[i],
            maximum(res[names[i]].density)*.8,
            text = L"\hat{\mu}=%$(round(means[i], digits=3))\ldots"
        )
        text!(
            axs[i],
            real[i],
            maximum(res[names[i]].density)*.75,
            text = L"\mu*=%$(real[i])"
        )
        text!(
            axs[i],
            minimum(res[names[i]].x),
            maximum(res[names[i]].density)*.2,
            text = L"\frac{1}{S}\sum{\frac{|x*-\hat{x}|}{x*}}=%$(round(relative_error[i], digits=3))\ldots"
        )
        axs[i].title = names[i]
        axislegend(axs[i], position = :lt)
    end
    fig
end

function show_parameters(par,md)
    ϕ, γ, δ, θ = par
    println("Structural parametes: ϕ = $ϕ, γ = $γ, $δ = δ, $θ = θ")
    σ = par[5]
    β = par[6:6+size(md.X,2)-1]
    println("Initial distribution parameters:  σ = $σ, β = $β")
    σⁱ_par = par[6+size(md.X,2):6+size(md.X,2)+size(md.s_data, 2)-1]
    println("Learning parameters:  σⁱ_par = $σⁱ_par")
    nothing
end

function gen_sample(par,md)
    N = size(md.X,1)
    T = length(md.costs)
    r = zeros(N,T)
    r_d = zeros(N,T)
    for i in 1:N
        md_sim = ModelData(
            zeros(Float64, 17),
            Vector{Float64}(md.costs),
            md.X,
            md.s_data,
            zeros(N),
            alg=Uniform()
        )
        a,b = patenthz(par,md_sim)
        r[i,:] .= a[i,:]
        r_d[i,:] .= b[i,:]
    end
    return r,r_d
end

function optimize_n(f,x0,n,lb=nothing,ub=nothing;alg=NelderMead(),options=Optim.Options())
    name = now().instant.periods.value
    mkdir("$name")
    lb = isnothing(lb) ? -Inf*ones(length(x0)) : lb
    ub = isnothing(ub) ? Inf*ones(length(x0)) : ub
    for i in 1:n
        res = optimize(
            f,
            lb,
            ub,
            rand.(x0),
            alg,
            options
        )
        save_object("$name/$i.jld2", res)
    end
end
