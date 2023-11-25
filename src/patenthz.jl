function patenthz(par, modeldata)
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
    r = modeldata.r
    r_d = modeldata.r_d
    X = modeldata.X
    obsolence = modeldata.obsolence
    x = modeldata.x
    ν = modeldata.ν
    nt = modeldata.nt
    hz = modeldata.hz
    T = length(hz)
    S = size(x, 1)
    
    r̄ = thresholds(par, modeldata)

    μ, σ = initial_shock_parametrisation(par, X)
    
    @inbounds begin
        r[:,1] .= x[:,1]# quantile.(LogNormal.(μ, σ), x[:,1])
        s = x[:,2:end]#-(log.(1 .-x[:,2:end]).*ϕ.^(1:T-1)'.*σⁱ.-γ)
        r_d[:,1] .= r[:,1] .≥ r̄[1]
    end
    o = obsolence .≤ θ
    
    inno_shock = mean(s, dims=1)
    
    idx_ranges = Int.(round.(LinRange(0, S, nt)))
    idx_ranges = [idx_ranges[i]+1:idx_ranges[i+1]
                    for i in 1:length(idx_ranges)-1]
    @Threads.threads for i in idx_ranges
        @inbounds for t=2:T
            # compute patent value at t by maximizing between learning shocks and depreciation
            r[i,t] .= o[i,t-1].*maximum(hcat(δ.*r[i,t-1], s[i,t-1]), dims=2) # concat as n×2 matrix and choose maximum in for each row
            # If patent wasn't active in t-1 it cannot be active in t
            r[i,t] .= r[i,t-1].*r_d[i,t-1]
            # Patent is kept active if its value exceed the threshold o.w. set to zero
            r_d[i,t] .= r[i,t] .≥ r̄[t]
            # ℓ[:,t] = likelihood(r, r̄, t, ν)
        end
    end

    patent_value = mean(r, dims=1)

    ℓ = cumprod(1 ./(1 .+exp.(-(r.-r̄')/ν)), dims=2)
    
    survive = vec(sum(ℓ', dims=2))
    ehz = modelhz(survive, S)

    @inbounds begin
        ehz[isnan.(ehz)] .= 0.
        err = ehz[2:end]-hz[2:end]
        err[isnan.(err)] .= 0
        W = Diagonal(sqrt.(survive[2:end]./S))
        fval = (err'*W*err)[1]
    end

    return (
        isnan(fval) ? Inf : fval,
        ehz,
        survive,
        inno_shock,
        patent_value,
        r
    )
end

function simulate_patenthz(par, modeldata)
    """
    simulate_patenthz(par::PatentModel, x)

    Function for simulating hazard rates for patent expirations.

        # Arguments:
        par::PatentModel: struct containing parameters for the distribution of patent exirations.
        x: random matrix distributed as U([0,1]) for drawing the values from LogNormal. Size N×T
        o: obsolence draw. Also U([0,1]) distributed random matrix. Size N×T-1. Used in both inner and outer loop.
    """
    ϕ, σⁱ, γ, δ, θ = par
    x = modeldata.x
    X = modeldata.X
    r = modeldata.r
    r_d = modeldata.r_d
    obsolence = modeldata.obsolence

    n, T = size(x)
    th = thresholds(par, modeldata)
    
    @assert length(th) == T "Dimension mismatch in time"

    μ, σ = initial_shock_parametrisation(par, X)

    r[:,1] .= x[:,1] # quantile.(LogNormal.(μ, σ), x[:,1])
    r_d[:,1] .= r[:,1] .≥ th[1]

    # size(z)=n×T⇒size(g(z))=T×n⇒size(g(z))'=n×T i.e. size(z)=n×T before and after this line (at least that is the intent)
    learning = x[:,2:end] # -(log.(1 .-x[:,2:end]).*ϕ.^(1:T-1)'.*σⁱ.-γ)
    o = obsolence .≤ θ
    # Computing patent values for t=2,…,T
    @inbounds for t=2:T
        # compute patent value at t by maximizing between learning shocks and depreciation
        r[:,t] .= o[:,t-1].*maximum(hcat(δ.*r[:,t-1], learning[:,t-1]), dims=2) # concat as n×2 matrix and choose maximum in for each row
        # If patent wasn't active in t-1 it cannot be active in t
        r[:,t] .= r[:,t-1].*r_d[:,t-1]
        # Patent is kept active if its value exceed the threshold o.w. set to zero
        r_d[:,t] .= r[:,t] .≥ th[t]
    end
    return (modelhz(sum(r_d, dims=1)', n), r, r_d)
end


likelihood(r, r̄, t, ν) = prod(1 ./(1 .+exp.(-(r[:,1:t-1].-r̄[1:t-1]')./ν)),dims=2).*1 ./(1 .+exp.((r[:,t].-r̄[t])./ν))

function log_norm_parametrisation(par, T)
    ϕ, σⁱ, γ = par

    # Conversion of mean and variance for log normal distribution according to the normal specification
    e_mean = ϕ.^(0:(T-1))*σⁱ*(1-γ)
    e_var = e_mean.^2
    # e_var = (ϕ.^(1:T)*σⁱ).^2

    μ = 2*log.(e_mean)-1/2*log.(e_mean.^2+e_var)
    σ = sqrt.(-2*log.(e_mean)+log.(e_var+e_mean.^2))

    return (μ, σ)
end

function initial_shock_parametrisation(par, X)
    σ = par[6]
    β = par[7:end]

    N = size(X,1)
    
    μ = hcat(ones(N),X)*β

    return (μ, σ)
end
