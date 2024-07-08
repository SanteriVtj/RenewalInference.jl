function thresholds(par, modeldata, x, obsolence)
    ϕ, γ, δ, θ = par
    c = modeldata.costs
    ngrid = modeldata.ngrid
    nt = modeldata.nt
    
    N, T, S = size(x)
    
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
    o = obsolence .≤ θ

    chunks = Iterators.partition(1:S, S÷nt > 0 ? S÷nt : 1)
    tasks = map(chunks) do chunk
        Threads.@spawn sim_total(chunk, par, modeldata, x, o, VT, r̄T, r1)
    end

    tsk = fetch.(tasks)
    rds = convert.(Float64, reduce(hcat, tsk))
    r̄ = sum(rds, dims=2)./S
    return r̄
end

function sim_total(chunk, par, modeldata, x, o, VT, r̄T, r1)
    β = modeldata.β
    N, T, S = size(x)
    ϕ, γ, δ, θ = par
    c = modeldata.costs
    ngrid = modeldata.ngrid
    Vtot = zeros(eltype(par), T, ngrid)
    r̄tot = zeros(eltype(par), T)

    @inbounds for s in chunk
        V = zeros(eltype(par), T, ngrid)
        V[T,:] .= VT
        r̄ = zeros(eltype(par), T)
        r̄[T] = r̄T
        for t=T-1:-1:1
            # Allocation for temp variables
            interp = linear_interpolation(
                r1,
                V[t+1, :], 
                extrapolation_bc=Line()
            )
            V[t,:] = r1'.-c[t].+β.*mean(interp.(o[s]*max.(x[:,t,s],δ*r1')), dims=1)
            # Gather positive values
            idx = findfirst(V[t,:].>zero(eltype(V)))
            m_idx = max(idx-1,1)
            r̄[t] = (idx == 1)  ? 0. : (r1[m_idx]*V[t,idx]-r1[idx]*V[t,m_idx])/(V[t,idx]-V[t,m_idx])
            V[t,:] = maximum([V[t,:] zeros(ngrid)], dims=2)
        end
        Vtot+=V
        r̄tot+=r̄
    end

    return r̄tot
end