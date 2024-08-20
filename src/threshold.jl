function thresholds(par, md, ma)
    ϕ, γ, δ, θ = par
    c = md.costs
    ngrid = md.ngrid
    β = md.β
    
    
    T = length(md.hz)
    N = size(md.X,1)
    S = length(md.x)
    
    r1 = collect(LinRange(0, maximum(c)+maximum(c)/ngrid, ngrid))

    # Compute values for t=T i.e. the last period from which the backwards induction begins
    @inbounds begin 
        ma.V[T,:] .= vec(r1' .- c[T])
        idx = findfirst(ma.V[T,:].>zero(eltype(par)))
        m_idx = max(idx-1,1)

        ma.r̄[T] = (r1[m_idx]*ma.V[T,idx]-r1[idx]*ma.V[T,m_idx])/
            (ma.V[T,idx]-ma.V[T,m_idx]);
        ma.V[T,:] .= max(ma.V[T,:], zeros(ngrid))
    end

    o = md.obsolence.≤θ

    @inbounds for t=T-1:-1:1
        # Allocation for temp variables
        interp = linear_interpolation(
            r1,
            ma.V[t+1, :],
            extrapolation_bc=Line()
        )

        # Compute value functions for each individual patent i.e. the Bellman
        ma.V[t,:] .= r1.-c[t].+β*mean(
            interp.(
                o[t].*max.(invF(md.x[t], t, ϕ, ma.σⁱ, γ),δ*r1')
            ), 
            dims=1
        )'
        # Gather positive values
        idx = findfirst(ma.V[t,:].>zero(eltype(ma.V)))
        m_idx = max(idx-1,1)
        ma.r̄[t] = (idx == 1)  ? 0. : (r1[m_idx]*ma.V[t,idx]-r1[idx]*ma.V[t,m_idx])/(ma.V[t,idx]-ma.V[t,m_idx])
        ma.V[t,:] .= max.(ma.V[t,:], zero(eltype(ma.V)))
    end
    
    # Gather positive values
    idx = findfirst(ma.V[1,:].>zero(eltype(ma.V)))
    m_idx = max(idx-1,1)
    ma.r̄[1] = (idx == 1)  ? 0. : (r1[m_idx]*ma.V[1,idx]-r1[idx]*ma.V[1,m_idx])/(ma.V[1,idx]-ma.V[1,m_idx])
    ma.V[1,:] .= max.(ma.V[1,:], zero(eltype(ma.V)))
end
