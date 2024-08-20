function patenthz(rrs::RRS, par, md, ma)
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

    ma.σⁱ .= md.s_data*par[6+size(md.X,2):6+size(md.X,2)+size(md.s_data, 2)-1]
    @inline thresholds(par, md, ma)

    # Compute initial shock parameters
    σ = par[5]
    β = par[6:6+size(md.X,2)-1]
    ma.μ .= md.X*β

    rrs.r[:,1] .= quantile.(LogNormal.(ma.μ, σ), md.x[1]) # shocks[:,1,s]
    rrs.r_d[:,1] .= rrs.r[:,1] .≥ ma.r̄[1]
    rrs.r[:,1] .= rrs.r[:,1].*rrs.r_d[:,1]

    o = md.obsolence.≤θ
    @inbounds for t=2:T
        # compute patent value at t by maximizing between learning shocks and depreciation
        rrs.r[:,t] .= o[t-1].*max(δ.*rrs.r[:,t-1], invF(md.x[t], t, ϕ, ma.σⁱ, γ)) # concat as n×2 matrix and choose maximum in for each row
        # If patent wasn't active in t-1 it cannot be active in t
        rrs.r[:,t] .= rrs.r[:,t].*rrs.r_d[:,t-1]
        # Patent is kept active if its value exceed the threshold o.w. set to zero
        rrs.r_d[:,t] .= rrs.r[:,t] .> ma.r̄[t]
    end
end

invF(z, t, ϕ, σⁱ, γ) = @. -(log(1-z)*ϕ^(t-1)*σⁱ-γ)

function simulate(par, md; S=1000, alg=QuasiMonteCarlo.HaltonSample(), shifting=Shift(), nt=Threads.nthreads())
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
            md_copy = deepcopy(md)
            ma = MemAlloc(
                zeros(eltype(par),N),
                zeros(eltype(par),T,md.ngrid),
                zeros(eltype(par),T),
                zeros(eltype(par),N)
            )
            rrs = RRS(
                zeros(eltype(par),N,T),
                zeros(eltype(par),N,T)
            )
            for s in chunk
                # Replace the sample with RQMC sample
                md_copy.x[:,:] .= randomize(QuasiMonteCarlo.sample(T,1,alg), shifting)
                md_copy.obsolence[:,:] .= randomize(QuasiMonteCarlo.sample(T-1,1,alg), shifting)
                # Simulate the DGP once
                @inline patenthz(rrs,par,md_copy,ma)
                # Save results
                r.+=rrs.r
                # keys = findfirst.(eachrow(rrs.r_d.==0))
                # keys[isnothing.(keys)] .= T
                # r_d[CartesianIndex.(zip(1:N,keys))].+=1
                r_d.+=rrs.r_d
                # @show s,unique(findfirst.(eachrow(rrs.r_d.==0)))
            end
            (r/S, r_d)
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
    # Create the deviation matrix for individual hazard rates
    hz[CartesianIndex.(zip(convert.(Int, md.renewals),1:17))].-=1
    # Compute aggregate errors for each age cohort
    ind_err = diag(hz.*W*hz')
    # return the total error as fval
    return ind_err'*ind_err
end