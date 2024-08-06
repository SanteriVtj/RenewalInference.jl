function patenthz(rrs::RRS, par, md)
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
    hz = md.hz
    T = length(hz)
    N = size(md.X,1)
    S = length(md.x)

    σⁱ = hcat(ones(eltype(par), N), md.s_data)*par[6+size(md.X,2)+1:6+size(md.X,2)+1+size(md.s_data, 2)]
    r̄ = @inline thresholds(par, md, σⁱ)

    rrs.r[:,1] .= quantile.(LogNormal.(initial_shock_parametrisation(par, md.X)...), md.x[1]) # shocks[:,1,s]
    rrs.r_d[:,1] .= rrs.r[:,1] .≥ r̄[1]
    rrs.r[:,1] .= rrs.r[:,1].*rrs.r_d[:,1]

    o = md.obsolence.≤θ
    @inbounds for t=2:T
        # compute patent value at t by maximizing between learning shocks and depreciation
        rrs.r[:,t] .= o[t-1].*max(δ.*rrs.r[:,t-1], invF(md.x[t], t, ϕ, σⁱ, γ)) # concat as n×2 matrix and choose maximum in for each row
        # If patent wasn't active in t-1 it cannot be active in t
        rrs.r[:,t] .= rrs.r[:,t].*rrs.r_d[:,t-1]
        # Patent is kept active if its value exceed the threshold o.w. set to zero
        rrs.r_d[:,t] .= rrs.r[:,t] .> r̄[t]
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

function simulate(rrs, par, md; S=1000, alg=QuasiMonteCarlo.HaltonSample(), shifting=Shift(), nt=Threads.nthreads())
    T = length(md.hz)
    N = size(md.X,1)
    # Initialize memory
    r = zeros(eltype(par),N,T)
    r_d = zeros(eltype(par),N,T)
    n_part = div(S,nt)
    n_part = n_part == 0 ? 1 : n_part
    chunks = Iterators.partition(1:S, n_part)
    tasks = map(chunks) do chunk
        Threads.@spawn begin
            # The simulation loop. Repeats DGP S times for S different realizations of RQMC 
            for s in chunk
                # Replace the sample with RQMC sample
                md.x[:,:] .= randomize(QuasiMonteCarlo.sample(T,1,alg), shifting)
                md.obsolence[:,:] .= randomize(QuasiMonteCarlo.sample(T-1,1,alg), shifting)
                # Simulate the DGP once
                @inline patenthz(rrs,par,md)
                # Save results
                r.+=rrs.r
                r_d.+=rrs.r_d
            end
            (r, r_d)
        end
    end 
    fetch.(tasks)[1]
end


function fval(rrs,par,md;S=1000,nt=Threads.nthread())
    T = length(md.costs)
    @assert all((md.renewals .≤ T).&(1 .≤ md.renewals)) "Given renewals must be within the support of data."
    N = size(md.X, 1)
    # Run the simulation
    r,r_d=simulate(rrs, par, md, S=S, nt=nt)
    # Construct individual patent simulation weighting matrix for the W-norm
    W = sqrt.(r_d/S)'
    # Compute simulation hazard rates
    hz = reduce(hcat, modelhz.(eachrow(r_d),S))
    # Create the deviation matrix for individual hazard rates
    hz[CartesianIndex.(zip(convert.(Int, md.renewals),1:17))].-=1
    # Compute aggregate errors for each age cohort
    ind_err = diag(hz.*W*hz')
    # return the total error as fval
    return ind_err'*ind_err
end