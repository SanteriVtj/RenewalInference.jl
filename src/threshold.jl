function thresholds(par, c, z, o, β, inventor_data; ngrid=500, n_threads=Threads.nthreads())
    _, _, δ, θ = par
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

    μ, σ = log_norm_parametrisation(par, inventor_data, T)

    z = exp.(z.*σ'.+μ')
    o = o .≤ θ
    
    # Create data partition for the simulation-wise threading
    thread_partition = collect(1:(ngrid ÷ n_threads):ngrid)
    thread_partition[end] = ngrid

    @inbounds for t=T-1:-1:1
        # Allocation for temp variables
        temp1 = repeat(δ.*r1', N)
        temp2 = repeat(z[:,t],1,ngrid)
        temp3 = repeat(o[:,t],1,ngrid)
        # Loop through the data partitions and assign new thread for each
        Threads.@threads for i in eachindex(thread_partition)[2:end]
            # Create ranges to assaing each part of the simulation to correct row in value function representation V
            col_range = thread_partition[i-1]:thread_partition[i]
            col_range_len = 1:(col_range.stop-col_range.start+1)

            # Interpolate the value function
            interp = linear_interpolation(
                r1[col_range],
                V[t+1, col_range], 
                extrapolation_bc=Line()
            )
            temp4 = interp.(dropdims(temp3[:,col_range].*maximum([temp1[:,col_range];;;temp2[:,col_range]], dims=3), dims=3))
            temp5 = 1/N.*ones(1,N)*temp4[:,col_range_len]
            # Compute patent values
            V[t,col_range] = r1[col_range]'.-c[t].+β.*temp5[:,col_range_len]
        end
        # Gather positive values
        idx = findfirst(V[t,:].>zero(eltype(V)))
        r̄[t] = (idx == 1) | isnothing(idx) ? 0. : (r1[idx-1]*V[t,idx]-r1[idx]*V[t,idx-1])/(V[t,idx]-V[t,idx-1])
        V[t,:] = maximum([V[t,:] zeros(size(V[t,:]))], dims=2)
    end
    return r̄
end
