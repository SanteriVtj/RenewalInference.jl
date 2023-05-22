function simulate_normhz(par, d=Nothing, seed=123, 
    z=100, lb=0, ub=1, 
    sample_method::QuasiMonteCarlo.SamplingAlgorithm=LowDiscrepancySample(2))
    """
    simulate_normhz(par, d=Nothing, seed=123, z=100, lb=0, ub=1, sample_method=SobolSample())

        Function to generate very simple hazard rate data for normally distributed data. Generates N(μ,σ) distributed sample and copies every n rows to |c| columns to generate n×|c| matrix. then applies cost vector c to test how "long" each row would "survive".
        # Arguments
    """
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

function normhz(x::Vector{T}, y::Vector{T}, z::Vector{T}, par::NormModel) where T<:Number
    μ=x[1];σ=x[2];
    ν=par.ν;n=par.n;c=par.c;

    z = μ.+z*σ
    
    temp = repeat(z,1,length(c))
    ℓ = cumprod(1 ./(1 .+exp.(-(temp.-c')/ν)), dims=2)
    survive = sum(ℓ,dims=1)
    ℓ = n.-survive
    ℓ = ℓ.-vcat(0, ℓ[1:length(ℓ)-1])'
    ŷ = (ℓ ./ vcat(n, survive[1:end-1])')'

    ((y[1:end-1].-ŷ)'*(y[1:end-1].-ŷ))[1]
end