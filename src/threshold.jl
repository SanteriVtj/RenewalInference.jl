function thresholds(par, c, z, o, β; ngrid=500)
    ϕ, σⁱ, γ, δ, θ = par
    T = length(c)
    N = length(z)

    # increment = 5
    # r_up = maximum(c)+increment
    # r1 = collect(0:increment:r_up) # 
    r1 = expm1.(range(0, log1p(maximum(c)), length=ngrid-1))
    append!(r1, 2*r1[end]-r1[end-1])
    # ngrid = length(r1)
    
    V = zeros(eltype(par), T, ngrid)
    r̄ = zeros(eltype(par), T)

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
    
    thread_partition = collect(1:(ngrid ÷ Threads.nthreads()):ngrid)
    thread_partition[end] = ngrid

    @inbounds for t=T-1:-1:1
        temp1 = repeat(δ.*r1', N)
        temp2 = repeat(z[:,t],1,ngrid)
        temp3 = repeat(o[:,t],1,ngrid)
        Threads.@threads for i in eachindex(thread_partition)[2:end]
            col_range = thread_partition[i-1]:thread_partition[i]
            col_range_len = 1:(col_range.stop-col_range.start+1)

            interp = linear_interpolation(
                r1[col_range],
                V[t+1, col_range], 
                extrapolation_bc=Line()
            )
            temp4 = interp.(dropdims(temp3[:,col_range].*maximum([temp1[:,col_range];;;temp2[:,col_range]], dims=3), dims=3))
            temp5 = 1/N.*ones(1,N)*temp4[:,col_range_len]
            V[t,col_range] = r1[col_range]'.-c[t].+β.*temp5[:,col_range_len]
        end
        idx = findfirst(V[t,:].>zero(eltype(V)))
        r̄[t] = (idx == 1) | isnothing(idx) ? 0. : (r1[idx-1]*V[t,idx]-r1[idx]*V[t,idx-1])/(V[t,idx]-V[t,idx-1])
        V[t,:] = maximum([V[t,:] zeros(size(V[t,:]))], dims=2)
    end
    r̄
end
