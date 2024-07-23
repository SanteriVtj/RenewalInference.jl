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
    hz = modeldata.hz
    T = length(hz)
    N = size(modeldata.X,1)
    s_data = modeldata.s_data
    X = modeldata.X
    x = modeldata.x
    obsolence = modeldata.obsolence
    nt = modeldata.nt
    S = length(x)

    σⁱ = hcat(ones(eltype(par), N), s_data)*par[6+size(X,2)+1:6+size(X,2)+1+size(s_data, 2)]
    
    r̄ = thresholds(par, modeldata, σⁱ)

    chunks = Iterators.partition(1:S, S÷nt > 0 ? S÷nt : 1)
    tasks = map(chunks) do chunk
        Threads.@spawn @inline patent_valu_total(chunk, par, modeldata, r̄, σⁱ)
    end
    # Get all of the results as matrices containing the mean values and renewal decisions
    val = fetch.(tasks)
    rtot = [val[i][1] for i in eachindex(val)]
    r_dtot = [val[i][2] for i in eachindex(val)]
    r = convert(Matrix{eltype(par)}, reduce(+, rtot)./S)
    r_d = convert(Matrix{eltype(par)}, reduce(+, r_dtot)./S)

    # Compute hazard rates based on the simulations 
    all_hz = reduce(hcat, modelhz.(eachrow(r_d*S), S))'
    all_hz[findall(isnan.(all_hz))] .= 0
    
    if modeldata.controller.simulation
        return (r_d, r, all_hz)
    else
        data_stopping = modeldata.renewals
        data_stopping = min.(data_stopping, T)
        data_stopping = max.(data_stopping, 1)
        hzd = zeros(eltype(par), N, T)
        hzd[CartesianIndex.(1:N, Int.(data_stopping))] .= 1
        err = abs.(all_hz.-hzd)
        err = diag(err'*err)
        fval = err'*err
        
        return fval
    end
end

likelihood(r, r̄, t, ν) = prod(1 ./(1 .+exp.(-(r[:,1:t-1].-r̄[1:t-1]')./ν)),dims=2).*1 ./(1 .+exp.((r[:,t].-r̄[t])./ν))

function initial_shock_parametrisation(par, X)
    σ = par[5]
    β = par[6:6+size(X,2)]

    N = size(X,1)
    
    μ = hcat(ones(eltype(par),N),X)*β
    
    return (μ, σ)
end

invF(z, t, ϕ, σⁱ, γ) = @. -(log(1-z)*ϕ^(t-1)*σⁱ-γ)

function patent_valu_total(chunk, par, modeldata, r̄, σⁱ)
    ϕ, γ, δ, θ = par
    T = length(modeldata.hz)
    N = size(modeldata.X,1)
    rtot = zeros(eltype(par), N, T)
    r_dtot = zeros(eltype(par), N, T)
    @inbounds for s in chunk
        r = zeros(eltype(par), N, T)
        r_d = zeros(eltype(par), N, T)
        r[:,1] .= quantile.(LogNormal.(initial_shock_parametrisation(par, modeldata.X)...), modeldata.x[s]) # shocks[:,1,s]
        r_d[:,1] .= r[:,1] .≥ r̄[1]
        r[:,1] .= r[:,1].*r_d[:,1]

        @inbounds for t=2:T
            o = modeldata.obsolence[t-1,s]≤θ
            # compute patent value at t by maximizing between learning shocks and depreciation
            r[:,t] .= o[t-1,s].*max(δ.*r[:,t-1], invF(modeldata.x[s], t, ϕ, σⁱ, γ)) # concat as n×2 matrix and choose maximum in for each row
            # If patent wasn't active in t-1 it cannot be active in t
            r[:,t] .= r[:,t].*r_d[:,t-1]
            # Patent is kept active if its value exceed the threshold o.w. set to zero
            r_d[:,t] .= r[:,t] .> r̄[t]
            # ℓ[:,t] = likelihood(r, r̄, t, ν)
        end

        rtot+=r
        r_dtot+=r_d
    end
    return (rtot, r_dtot)
end