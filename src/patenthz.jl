function patenthz(
    par, hz, initial_shock, obsolence, costs;
    opt=true, ν=2, β=.95
)
    """
    # Arguments
    par::Vector{Float64}: vector containing parameters for the distribution of patent exirations.
    hz: Empirical hazard rates.
    s: Simulation draw.
    o: Obsolence draw.
    c: Renewal costs for patents.
    z: Initial shocks for inner loop simulation. Random or quasirandom draw of size N.
    o2: obsolence shocks in inner loop. Random or quasirandom draw of size N×T.
    """
    ϕ, σⁱ, γ, δ, θ = par

    S = length(initial_shock)
    T = length(hz)
    r = zeros(eltype(par), S,T)
    
    r_d = falses(S,T) # Equivalent of zeros(UInt8,n,m), but instead of UInt8 stores elements as single bits
    r̄ = thresholds(par, costs, initial_shock, obsolence, β)

    μ, σ = log_norm_parametrisation(par, T)

    s = exp.(initial_shock.*σ'.+μ')
    inno_shock = mean(s, dims=1)
    @inbounds begin
        r[:,1] = s[:,1]
        r_d[:,1] = r[:,1] .≥ r̄[1]
    end
    o = obsolence .≤ θ

    @inbounds for t=2:T
        # compute patent value at t by maximizing between learning shocks and depreciation
        r[:,t] .= o[:,t-1].*maximum(hcat(δ.*r[:,t-1], s[:,t-1]), dims=2) # concat as n×2 matrix and choose maximum in for each row
        # If patent wasn't active in t-1 it cannot be active in t
        r[r_d[:,t-1] .== 0,t] .= 0
        # Patent is kept active if its value exceed the threshold o.w. set to zero
        r_d[:,t] .= r[:,t] .≥ r̄[t]
        # ℓ[:,t] = likelihood(r, r̄, t, ν)
    end

    patent_value = mean(r, dims=1) 

    ℓ = cumprod(1 ./(1 .+exp.(-(r.-r̄')/ν)), dims=2)
    
    survive = vec(sum(ℓ', dims=2))
    ehz = modelhz(survive, S)
    @inbounds err = ehz[2:end]-hz[2:end]
    # W = Diagonal(sqrt.(survive[2:end]./S))
    W = I
    return (
        (err'*W*err)[1],
        ehz,
        survive,
        inno_shock,
        patent_value
    )
end

function simulate_patenthz(par, x, o, c, ishocks,
    ν=2, β=.95
)
    """
    simulate_patenthz(par::PatentModel, x)

    Function for simulating hazard rates for patent expirations.

        # Arguments:
        par::PatentModel: struct containing parameters for the distribution of patent exirations.
        x: random matrix distributed as U([0,1]) for drawing the values from LogNormal. Size N×T
        o: obsolence draw. Also U([0,1]) distributed random matrix. Size N×T-1. Used in both inner and outer loop.
    """
    ϕ, σⁱ, γ, δ, θ = par
    
    n, T = size(x)
    q = @view x[:,1]
    z = @view x[:,2:end]
    th = thresholds(par, c, ishocks, o, β)
    
    @assert length(th) == T "Dimension mismatch in time"

    μ, σ = log_norm_parametrisation(par, T)

    # Zero matrix for patent values and active patent periods
    r = zeros(n,T)
    r_d = zeros(n,T)
    obsolence = zeros(n,T-1)

    r[:,1] .= quantile.(LogNormal(μ[1], σ[1]), q)
    r_d[:,1] .= r[:,1] .≥ th[1]

    # size(z)=n×T⇒size(g(z))=T×n⇒size(g(z))'=n×T i.e. size(z)=n×T before and after this line (at least that is the intent)
    learning = quantile.(LogNormal.(μ[2:end], σ[2:end]), z')'
    obsolence .= o .≤ θ

    # Computing patent values for t=2,…,T
    @inbounds for t=2:T
        # compute patent value at t by maximizing between learning shocks and depreciation
        r[:,t] .= obsolence[:,t-1].*maximum(hcat(δ.*r[:,t-1], learning[:,t-1]), dims=2) # concat as n×2 matrix and choose maximum in for each row
        # If patent wasn't active in t-1 it cannot be active in t
        r[r_d[:,t-1] .== 0,t] .= 0
        # Patent is kept active if its value exceed the threshold o.w. set to zero
        r_d[:,t] .= r[:,t] .≥ th[t]
    end

    (computehz(sum(r_d, dims=2)), r, r_d)
end


likelihood(r, r̄, t, ν) = prod(1 ./(1 .+exp.(-(r[:,1:t-1].-r̄[1:t-1]')./ν)),dims=2).*1 ./(1 .+exp.((r[:,t].-r̄[t])./ν))

function log_norm_parametrisation(par, T)
    ϕ, σⁱ, γ, δ, θ = par

    # Conversion of mean and variance for log normal distribution according to the normal specification
    e_mean = ϕ.^(1:T)*σⁱ*(1-γ)
    e_var = e_mean.^2
    
    e_mean[e_mean .≤ 0] .= 0

    μ = 2*log.(e_mean)-1/2*log.(e_mean.^2+e_var)
    σ = sqrt.(-2*log.(e_mean)+log.(e_var+e_mean.^2))

    (μ, σ)
end
