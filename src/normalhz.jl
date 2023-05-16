using Distributions, StatsBase

function simulate_normhz(par, d=Normal(), seed=123)
    Random.seed!(seed)
    n=par.n;c=par.c

    x = repeat(rand(d, n), 1, length(c))
    x = sum(x.-c' .≥ 0, dims=2)

    x1 = countmap(x)
    x1 = []

    return x
end

function normhz()
    return
end