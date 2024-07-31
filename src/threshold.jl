function thresholds(par, modeldata, σⁱ)
    ϕ, γ, δ, θ = par
    c = modeldata.costs
    ngrid = modeldata.ngrid
    nt = modeldata.nt
    β = modeldata.β
    
    
    T = length(modeldata.hz)
    N = size(modeldata.X,1)
    S = length(modeldata.x)
    
    r1 = collect(LinRange(0, maximum(c)+maximum(c)/ngrid, ngrid))

    # Compute values for t=T i.e. the last period from which the backwards induction begins
    @inbounds begin 
        VT = vec(r1' .- c[T])
        idx = findfirst(VT.>zero(eltype(par)))
        m_idx = max(idx-1,1)

        r̄T = (r1[m_idx]*VT[idx]-r1[idx]*VT[m_idx])/
            (VT[idx]-VT[m_idx]);
        VT .= max(VT, zeros(ngrid))
    end

    # Pre-allocate memory
    Vtot = zeros(eltype(par), T, ngrid)
    r̄tot = zeros(eltype(par), T)

    V = zeros(eltype(par), T, ngrid)
    r̄ = zeros(eltype(par), T)
    
    @inbounds for s in 1:S
        V[T,:] .= VT
        r̄[T] = r̄T
        for t=T-1:-1:2
            o = modeldata.obsolence[t,s]≤θ
            # Allocation for temp variables
            interp = linear_interpolation(
                r1,
                V[t+1, :], 
                extrapolation_bc=Line()
            )

            V[t,:] = r1'.-c[t].+β.*mean(interp.(o*max.(invF(modeldata.x[s], t, ϕ, σⁱ, γ),δ*r1')), dims=1)
            # Gather positive values
            idx = findfirst(V[t,:].>zero(eltype(V)))
            m_idx = max(idx-1,1)
            r̄[t] = (idx == 1)  ? 0. : (r1[m_idx]*V[t,idx]-r1[idx]*V[t,m_idx])/(V[t,idx]-V[t,m_idx])
            V[t,:] = maximum([V[t,:] zeros(ngrid)], dims=2)
        end
        o = modeldata.obsolence[1,s]≤θ
        # Allocation for temp variables
        interp = linear_interpolation(
            r1,
            V[2, :], 
            extrapolation_bc=Line()
        )
        V[1,:] = r1'.-c[1].+β.*mean(interp.(o*max.(quantile.(LogNormal.(initial_shock_parametrisation(par, modeldata.X)...), modeldata.x[s]),δ*r1')), dims=1)
        # Gather positive values
        idx = findfirst(V[1,:].>zero(eltype(V)))
        m_idx = max(idx-1,1)
        r̄[1] = (idx == 1)  ? 0. : (r1[m_idx]*V[1,idx]-r1[idx]*V[1,m_idx])/(V[1,idx]-V[1,m_idx])
        V[1,:] = maximum([V[1,:] zeros(ngrid)], dims=2)

        Vtot+=V
        r̄tot+=r̄
    end

    r̄ = sum(r̄tot, dims=2)./S
    return r̄
end
