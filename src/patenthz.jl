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
    ϕ, γ, δ, θ = par
    obsolence = modeldata.obsolence
    x = modeldata.x
    ν = modeldata.ν
    hz = modeldata.hz
    T = length(hz)
    S = size(x, 1)

    r = zeros(eltype(par), S, T)
    r_d = zeros(eltype(par), S, T)
    r̄ = thresholds(par, modeldata)

    @inbounds begin
        r[:,1] .= x[:,1]# quantile.(LogNormal.(μ, σ), x[:,1])
        s = x[:,2:end]#-(log.(1 .-x[:,2:end]).*ϕ.^(1:T-1)'.*σⁱ.-γ)
        r_d[:,1] .= r[:,1] .≥ r̄[1]
    end
    o = obsolence .≤ θ
    
    inno_shock = mean(s, dims=1)
    
    @Threads.threads for i in 1:length(eachrow(r))
    @inbounds for t=2:T
        # compute patent value at t by maximizing between learning shocks and depreciation
        r[i,t] = o[i,t-1].*max(δ.*r[i,t-1], s[i,t-1]) # concat as n×2 matrix and choose maximum in for each row
        # If patent wasn't active in t-1 it cannot be active in t
        r[i,t] = r[i,t].*r_d[i,t]
        # Patent is kept active if its value exceed the threshold o.w. set to zero
        r_d[i,t] = r[i,t] .> r̄[t]
        # ℓ[:,t] = likelihood(r, r̄, t, ν)
    end
    end
    ℓ = cumprod(1 ./(1 .+exp.(-(r.-r̄')/ν)), dims=2)
    
    survive = vec(sum(ℓ', dims=2))
    ehz = modelhz(survive, S)

    if modeldata.controller.simulation
        modeldata.hz[:] .= modelhz(sum(r_d, dims=1)', S)
        return (
            modelhz(sum(r_d, dims=1)', S),
            r, 
            r_d
        )
    end

    @inbounds begin
        ehz[isnan.(ehz)] .= 0.
        err = ehz[2:end]-hz[2:end]
        err[isnan.(err)] .= 0
        W = Diagonal(sqrt.(survive[2:end]./S))
        fval = (err'*W*err)[1]
        fval = isnan(fval) ? Inf : fval
    end

    if modeldata.controller.debug
        return (
            fval,
            ehz,
            inno_shock,
            r,
            r_d,
            ℓ
        )
    elseif modeldata.controller.ae_mode
        return sum(r_d, dims=2)
    end

    return fval
end

likelihood(r, r̄, t, ν) = prod(1 ./(1 .+exp.(-(r[:,1:t-1].-r̄[1:t-1]')./ν)),dims=2).*1 ./(1 .+exp.((r[:,t].-r̄[t])./ν))

function initial_shock_parametrisation(par, X)
    σ = par[5]
    β = par[6:6+size(X,2)]

    N = size(X,1)
    
    μ = hcat(ones(N),X)*β
    
    return (μ, σ)
end
