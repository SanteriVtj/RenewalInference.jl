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
    ν = modeldata.ν
    hz = modeldata.hz
    T = length(hz)
    N = size(modeldata.X,1)
    s_data = modeldata.s_data
    X = modeldata.X

    μ, σ = initial_shock_parametrisation(par, X)
    
    obsolence = QuasiMonteCarlo.sample(N,T-1,modeldata.alg)'
    x = QuasiMonteCarlo.sample(N,T,modeldata.alg)'

    σⁱ = hcat(ones(eltype(par), N), s_data)*par[6+size(X,2)+1:6+size(X,2)+1+size(s_data, 2)]
    
    shocks = zeros(eltype(par), N, T)
    shocks[:,1] = quantile.(LogNormal.(μ, σ), x[:,1])
    shocks[:,2:end] = -(log.(1 .-x[:,2:end]).*ϕ.^(1:T-1)'.*σⁱ.-γ)
    
    r = zeros(eltype(par), N, T)
    r_d = zeros(eltype(par), N, T)
    # r̄ = thresholds(par, modeldata, shocks, obsolence)
    r̄ = modeldata.costs

    @inbounds begin
        r[:,1] .= shocks[:,1]# quantile.(LogNormal.(μ, σ), x[:,1])
        s = shocks[:,2:end]#-(log.(1 .-x[:,2:end]).*ϕ.^(1:T-1)'.*σⁱ.-γ)
        r_d[:,1] .= r[:,1] .≥ r̄[1]
        r[:,1] .= r[:,1].*r_d[:,1]
    end

    o = obsolence .≤ θ
    
    inno_shock = mean(s, dims=1)
    @Threads.threads for i in 1:length(eachrow(r))
        @inbounds for t=2:T
            # compute patent value at t by maximizing between learning shocks and depreciation
            r[i,t] = o[i,t-1].*max(δ.*r[i,t-1], s[i,t-1]) # concat as n×2 matrix and choose maximum in for each row
            # If patent wasn't active in t-1 it cannot be active in t
            r[i,t] = r[i,t].*r_d[i,t-1]
            # Patent is kept active if its value exceed the threshold o.w. set to zero
            r_d[i,t] = r[i,t] .> r̄[t]
            # ℓ[:,t] = likelihood(r, r̄, t, ν)
        end
    end
    ℓ = cumprod(1 ./(1 .+exp.(-(r.-r̄')/ν)), dims=2)
    survive = vec(sum(ℓ', dims=2))
    survive[survive.==zero(eltype(survive))] .= survive[survive.==zero(eltype(survive))].+1e-12
    
    ehz = modelhz(survive, N)
    
    if eltype(ehz)<:ForwardDiff.Dual
        ehz[findall(x->any(isnan.(x.partials)), ehz)] .= zero(eltype(ehz))
    end

    if modeldata.controller.ae_mode
        return ehz#sum(r_d, dims=2)
    elseif modeldata.controller.simulation
        modeldata.hz[:] .= modelhz(sum(r_d, dims=1)', N)
        return (
            ehz, # modelhz(sum(r_d, dims=1)', S),
            r, 
            r_d
        )
    end

    @inbounds begin
        ehz[isnan.(ehz)] .= zero(eltype(ehz))
        err = ehz[2:end]-hz[2:end]
        err[isnan.(err)] .= zero(eltype(err))
        w = sqrt.(survive[2:end]./N)
        if eltype(w)<:ForwardDiff.Dual
            w[findall(x->any(isnan.(x.partials)), w)] .= zero(eltype(w))
        end
        W = Diagonal(w)
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
    end

    return fval
end

likelihood(r, r̄, t, ν) = prod(1 ./(1 .+exp.(-(r[:,1:t-1].-r̄[1:t-1]')./ν)),dims=2).*1 ./(1 .+exp.((r[:,t].-r̄[t])./ν))

function initial_shock_parametrisation(par, X)
    σ = par[5]
    β = par[6:6+size(X,2)]

    N = size(X,1)
    
    μ = hcat(ones(eltype(par),N),X)*β
    
    return (μ, σ)
end
