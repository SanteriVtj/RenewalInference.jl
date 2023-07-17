function thresholds(par, c, z, o, β, increment=5)
    ϕ, σⁱ, γ, δ, θ = par
    T = length(c)
    N = length(z)

    r_up = maximum(c)+increment
    r1 = collect(0:increment:r_up)
    ngrid = length(r1)

    V = zeros(eltype(par), T, ngrid)
    r̄ = zeros(eltype(par), T)

    @inbounds V[T,:] = r1' .- c[T]
    idx = findfirst(V[T,:].>zero(eltype(V)))
    m_idx = maximum([idx-1,1])
    
    @inbounds r̄[T] = (r1[m_idx]*V[T,idx]-r1[idx]*V[T,m_idx])/
        (V[T,idx]-V[T,m_idx]);
    @inbounds V[T,:] = maximum(hcat(V[T,:], zeros(length(V[T,:]))), dims=2)

    μ, σ = log_norm_parametrisation(par, T)

    z = exp.(z.*σ'.+μ')
    o = o .≤ θ
    
    @inbounds for t=T-1:-1:1
        temp1 = repeat(δ.*r1', N)
        temp2 = repeat(z[:,t],1,ngrid)
        temp3 = repeat(o[:,t],1,ngrid)
        interp = linear_interpolation(r1, view(V, t+1, :), extrapolation_bc=Line())
        temp4 = interp.(dropdims(temp3.*maximum([temp1;;;temp2], dims=3), dims=3))
        temp5 = 1/N.*ones(1,N)*temp4
        V[t,:] = r1'.-c[t].+β.*temp5
        idx = findfirst(V[t,:].>zero(eltype(V)))
        r̄[t] = idx == 1 ? 0. : (r1[idx-1]*V[t,idx]-r1[idx]*V[t,idx-1])/(V[t,idx]-V[t,idx-1])
        V[t,:] = maximum([V[t,:] zeros(size(V[t,:]))], dims=2)
    end
    r̄
end
