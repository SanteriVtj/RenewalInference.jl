using Distributions, StatsBase, QuasiMonteCarlo, Random

function simulate_normhz(par, d=Nothing, seed=123, 
    z=100, lb=0, ub=1, sample_method=SobolSample())
    Random.seed!(seed)
    μ=par.μ;σ=par.σ;n=par.n;c=par.c
    if d == Nothing
        d = Normal(μ, σ)
    end

    x = repeat(rand(d, n), 1, length(c))
    x = sum(x.-c' .≥ 0, dims=2)

    x1 = countmap(x)
    x1 = [get(x1, i, Nothing) for i=0:length(c)]

    if Nothing ∈ x1
        throw(DomainError(Nothing))
    end

    y = sum(x1).-cumsum(x1)
    y = [sum(x1); y[1:length(y)-1]]
    y = x1 ./ y

    return (
            y,
            QuasiMonteCarlo.sample(z,lb,ub,sample_method)
        )
end

function normhz(x, par)
    μ=x[1];σ=x[2];
    ν=par.ν;n=par.n;c=par.c;y=par.y;z=par.z;

    z = μ+z*σ
    r_d = (z.-c).≥0
    
    ℓ = cumprod(1 ./(1+exp(-(z.-c)/ν)), dims=2)

    survive = sum(ℓ)

    ℓ = n.-survive
    ℓ = ℓ.-[0;ℓ[1:length(ℓ)-1]]

end