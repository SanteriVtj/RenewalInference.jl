function patenthz(par, md::ModelData)
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
    r = zeros(eltype(par), N, T)
    r_d = zeros(eltype(par), N, T)

    # ma.σⁱ .= md.s_data*par[6+size(md.X,2):6+size(md.X,2)+size(md.s_data, 2)-1]
    σⁱ = md.s_data*par[6+size(md.X,2):6+size(md.X,2)+size(md.s_data, 2)-1]
    r̄ = thresholds(par, md, σⁱ)

    # Compute initial shock parameters
    σ = par[5]
    β = par[6:6+size(md.X,2)-1]
    # ma.μ .= md.X*β
    μ = md.X*β

    r[:,1] .= quantile.(LogNormal.(μ, σ), md.x[1]) # shocks[:,1,s]
    r_d[:,1] .= r[:,1] .≥ r̄[1]
    r[:,1] .= r[:,1].*r_d[:,1]

    o = md.obsolence.≤θ
    @inbounds for t=2:T
        # compute patent value at t by maximizing between learning shocks and depreciation
        r[:,t] .= o[t-1].*max(δ.*r[:,t-1], invF(md.x[t], t, ϕ, σⁱ, γ)) # concat as n×2 matrix and choose maximum in for each row
        # If patent wasn't active in t-1 it cannot be active in t
        r[:,t] .= r[:,t].*r_d[:,t-1]
        # Patent is kept active if its value exceed the threshold o.w. set to zero
        r_d[:,t] .= r[:,t] .> r̄[t]
    end
    
    return r, r_d
end

invF(z, t, ϕ, σⁱ, γ) = @. -(log(1-z)*ϕ^(t-1)*σⁱ-γ)

function simulate(par, md; S=1000, alg=QuasiMonteCarlo.HaltonSample(), shifting=Shift(), nt=Threads.nthreads())
    T = length(md.hz)
    N = size(md.X,1)
    # Initialize memory
    rtot = zeros(eltype(par),N,T)
    r_dtot = zeros(eltype(par),N,T)
    n_part = div(S,nt)
    n_part = n_part == 0 ? 1 : n_part
    chunks = Iterators.partition(1:S, n_part)
    tasks = map(chunks) do chunk
        Threads.@spawn begin
            # The simulation loop. Repeats DGP S times for S different realizations of RQMC 
            md_copy = deepcopy(md)
            for s in chunk
                # Replace the sample with RQMC sample
                md_copy.x[:,:] .= randomize(QuasiMonteCarlo.sample(T,1,alg), shifting)
                md_copy.obsolence[:,:] .= randomize(QuasiMonteCarlo.sample(T-1,1,alg), shifting)
                # Simulate the DGP once
                r, r_d = patenthz(par,md_copy)
                # Save results
                rtot.+=r
                r_dtot.+=r_d
            end
            (rtot/S, r_dtot)
        end
    end
    fetch.(tasks)[1]
end


function fval(par,md;S=1000,nt=Threads.nthreads())
    T = length(md.costs)
    @assert all((md.renewals .≤ T).&(1 .≤ md.renewals)) "Given renewals must be within the support of data."
    N = size(md.X, 1)
    # Run the simulation
    r,r_d=simulate(par, md, S=S, nt=nt)
    # Construct individual patent simulation weighting matrix for the W-norm
    W = sqrt.(r_d/S)'
    # Compute simulation hazard rates
    hz = reduce(hcat, modelhz.(eachrow(r_d),S))
    return hz
    # Create the deviation matrix for individual hazard rates
    hz[CartesianIndex.(zip(convert.(Int, md.renewals),1:17))].-=1
    # Compute aggregate errors for each age cohort
    ind_err = diag(hz.*W*hz')
    # return the total error as fval
    return ind_err'*ind_err
end