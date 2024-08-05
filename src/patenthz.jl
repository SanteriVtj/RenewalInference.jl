function patenthz(rrs::RRS, par, modeldata)
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
    nt = modeldata.nt
    S = length(x)

    σⁱ = hcat(ones(eltype(par), N), s_data)*par[6+size(X,2)+1:6+size(X,2)+1+size(s_data, 2)]
    r̄ = @inline thresholds(par, modeldata, σⁱ)

    rrs.r[:,1] .= quantile.(LogNormal.(initial_shock_parametrisation(par, modeldata.X)...), modeldata.x[1]) # shocks[:,1,s]
    rrs.r_d[:,1] .= rrs.r[:,1] .≥ r̄[1]
    rrs.r[:,1] .= rrs.r[:,1].*rrs.r_d[:,1]

    o = modeldata.obsolence.≤θ
    @inbounds for t=2:T
        # compute patent value at t by maximizing between learning shocks and depreciation
        rrs.r[:,t] .= o[t-1].*max(δ.*rrs.r[:,t-1], invF(modeldata.x[t], t, ϕ, σⁱ, γ)) # concat as n×2 matrix and choose maximum in for each row
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

# function simulate(par, md; S=1000)
#     T = length(md.hz)
#     N = size(md.X,1)
    
#     return 
# end
