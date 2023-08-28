function _patenthz(x0, hz, c; β=.95, ν=2, N=200, T=17, alg=QuasiMonteCarlo.HaltonSample())
    obsolence = QuasiMonteCarlo.sample(N,T-1,alg)'
    ishock = quantile.(Normal(), QuasiMonteCarlo.sample(N÷2,1,alg)')
    ishock = [ishock;-ishock]
    return patenthz(
        x0,
        hz,
        ishock,
        obsolence,
        c,
        β=β,
        ν=ν
    )
end

function _simulate_patenthz(par, c; N=200, T=17, alg=QuasiMonteCarlo.HaltonSample())
    simulation_shocks = QuasiMonteCarlo.sample(N,T,alg)'
    obsolence = QuasiMonteCarlo.sample(N,T-1,alg)'
    ishock = quantile.(Normal(), QuasiMonteCarlo.sample(N,1,alg)')
    return simulate_patenthz(
        par,
        simulation_shocks,
        obsolence,
        c,
        ishock
    )
end

function _thresholds(par,c; β=.95, N=200, T=17, alg=QuasiMonteCarlo.HaltonSample())
    obsolence = QuasiMonteCarlo.sample(N,T-1,alg)'
    ishock = quantile.(Normal(), QuasiMonteCarlo.sample(N,1,alg)')
    return thresholds(par, c, ishock, obsolence, β)
end

function plot_paramdist(x, real; cols = 2)
    names = DataFrames.names(x)

    rows = mod(length(names), 2) == 0 ? length(names) ÷ cols : length(names) ÷ 2+1

    fig = Figure(resolution = (400*rows, 600*cols))
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
            text = L"\frac{1}{S}\sum{\frac{|x*-\hat{x}|}{\hat{x}}}=%$(round(relative_error[i], digits=3))\ldots"
        )
        axs[i].title = names[i]
        axislegend(axs[i], position = :lt)
    end
    fig
end