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
    T = length(md.costs)

    renewals = zeros(size(md.X,1))
    for i in 1:size(md.X,1)
        md_sim = ModelData(
            zeros(Float64, 17),
            md.costs,
            reshape(md.X[i,:],1,size(md.X,2)),
            reshape(md.s_data[i,:],1,size(md.s_data,2)),
            zeros(1),
            alg=Uniform()
        )
        r,r_d = simulate(par, md_sim,S=1)
        drop = findfirst(r_d .==0)
        renewals[i] = isnothing(drop) ? T :  drop.I[2]-1
    end
    return renewals
end