function thresholds(par, c, z, o, β; ngrid=500, n_threads=Threads.nthreads())
    ϕ, σⁱ, γ, δ, θ = par
    T = length(c)
    N = length(z)

    r1 = expm1.(range(0, log1p(maximum(c)), length=ngrid-1))
    append!(r1, 2*r1[end]-r1[end-1])
    
    V = zeros(eltype(par), T, ngrid)
    r̄ = zeros(eltype(par), T)

    # Compute values for t=T i.e. the last period from which the backwards induction begins
    @inbounds begin 
        V[T,:] = r1' .- c[T]
        idx = any(V[T,:].>zero(eltype(V))) ? findfirst(V[T,:].>zero(eltype(V))) : 1
        m_idx = maximum([idx-1,1])

        r̄[T] = (r1[m_idx]*V[T,idx]-r1[idx]*V[T,m_idx])/
            (V[T,idx]-V[T,m_idx]);
        V[T,:] = maximum(hcat(V[T,:], zeros(length(V[T,:]))), dims=2)
    end

    μ, σ = log_norm_parametrisation(par, T)

    z = exp.(z.*σ'.+μ')
    o = o .≤ θ

    @inbounds for t=T-1:-1:1
        # Allocation for temp variables
        temp1 = repeat(δ.*r1', N)
        temp2 = repeat(z[:,t],1,ngrid)
        temp3 = repeat(o[:,t],1,ngrid)
        interp = linear_interpolation(
            r1,
            V[t+1, :], 
            extrapolation_bc=Line()
        )
        temp4 = interp.(
            dropdims(
                temp3.*maximum([temp1;;;temp2], dims=3), 
                dims=3
                )
            )
        temp5 = 1/N.*ones(1,N)*temp4
        # Compute patent values
        V[t,:] = r1'.-c[t].+β.*temp5

        # Gather positive values
        idx = findfirst(V[t,:].>zero(eltype(V)))
        r̄[t] = (idx == 1) | isnothing(idx) ? 0. : (r1[idx-1]*V[t,idx]-r1[idx]*V[t,idx-1])/(V[t,idx]-V[t,idx-1])
        V[t,:] = maximum([V[t,:] zeros(size(V[t,:]))], dims=2)
    end
    
    return r̄
end
