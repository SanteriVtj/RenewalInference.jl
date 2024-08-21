function thresholds(par, md, σⁱ)
    ϕ, γ, δ, θ = par
    c = md.costs
    ngrid = md.ngrid
    β = md.β
    
    
    T = length(md.hz)
    N = size(md.X,1)
    S = length(md.x)
    V = zeros(N,md.ngrid)
    r̄ = zeros(T)
    
    r1 = collect(LinRange(0, maximum(c)+maximum(c)/ngrid, ngrid))

    # Compute values for t=T i.e. the last period from which the backwards induction begins
    @inbounds begin 
        V[T,:] .= vec(r1' .- c[T])
        idx = findfirst(V[T,:].>zero(eltype(par)))
        m_idx = max(idx-1,1)

        r̄[T] = (r1[m_idx]*V[T,idx]-r1[idx]*V[T,m_idx])/
            (V[T,idx]-V[T,m_idx]);
        V[T,:] .= max(V[T,:], zeros(ngrid))
    end

    o = md.obsolence.≤θ

    @inbounds for t=T-1:-1:1
        # Allocation for temp variables
        interp = linear_interpolation(
            r1,
            V[t+1, :],
            extrapolation_bc=Line()
        )

        # Compute value functions for each individual patent i.e. the Bellman
        V[t,:] .= r1.-c[t].+β*mean(
            interp.(
                o[t].*max.(invF(md.x[t], t, ϕ, σⁱ, γ),δ*r1')
            ), 
            dims=1
        )'
        # Gather positive values
        idx = findfirst(V[t,:].>zero(eltype(V)))
        m_idx = max(idx-1,1)
        r̄[t] = (idx == 1)  ? 0. : (r1[m_idx]*V[t,idx]-r1[idx]*V[t,m_idx])/(V[t,idx]-V[t,m_idx])
        V[t,:] .= max.(V[t,:], zero(eltype(V)))
    end
    
    # Gather positive values
    idx = findfirst(V[1,:].>zero(eltype(V)))
    m_idx = max(idx-1,1)
    r̄[1] = (idx == 1)  ? 0. : (r1[m_idx]*V[1,idx]-r1[idx]*V[1,m_idx])/(V[1,idx]-V[1,m_idx])
    V[1,:] .= max.(V[1,:], zero(eltype(V)))
    
    return r̄ 
end
