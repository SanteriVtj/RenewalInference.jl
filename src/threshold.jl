function thresholds(par, modeldata)
    ϕ, σⁱ, γ, δ, θ = par
    c = modeldata.costs
    x = modeldata.x
    obsolence = modeldata.obsolence
    X = modeldata.X
    β = modeldata.β
    ngrid = modeldata.ngrid
    nt = modeldata.nt
    V = @view modeldata.V[:,:]


    T = length(c)
    N, M = size(x)

    r1 = collect(range(0, maximum(c), length=ngrid-1))
    append!(r1, last(r1)+last(diff(r1)))
    
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

    # μ, σ = log_norm_parametrisation(par, T)
    μ, σ = initial_shock_parametrisation(par, X)

    x[:,1] .= quantile.(LogNormal.(μ, σ), x[:,1])
    x[:,2:end] .= -(log.(1 .-x[:,2:end]).*ϕ.^(1:T-1)'.*σⁱ.-γ)
    o = obsolence .≤ θ

    idx_ranges = Int.(round.(LinRange(0, N, nt)))
    idx_ranges = [idx_ranges[i]+1:idx_ranges[i+1]
                    for i in 1:length(idx_ranges)-1]
    @inbounds for t=T-1:-1:1
        # Allocation for temp variables
        temp1 = repeat(δ.*r1', N)
        temp2 = repeat(x[:,t],1,ngrid)
        temp3 = repeat(o[:,t],1,ngrid)
        interp = linear_interpolation(
            r1,
            V[t+1, :], 
            extrapolation_bc=Line()
        )
        temp4 = _calctemp4(temp1,temp2,temp3,interp,idx_ranges)
        temp5 = 1/N.*ones(1,N)*temp4
        # Compute patent values
        V[t,:] = r1'.-c[t].+β.*temp5

        # Gather positive values
        idx = findfirst(V[t,:].>zero(eltype(V)))
        r̄[t] = (idx == 1) | isnothing(idx) ? 0. : (r1[idx-1]*V[t,idx]-r1[idx]*V[t,idx-1])/(V[t,idx]-V[t,idx-1])
        V[t,:] .= maximum([V[t,:] zeros(size(V[t,:]))], dims=2)
    end
    
    return r̄
end

function _calctemp4(temp1,temp2,temp3,interp,idx_ranges)
    temp4 = zeros(size(temp3))
    Threads.@threads for i in idx_ranges
        @views temp4[i,:] .= interp.(
            temp3[i,:].*max.(temp1[i,:], temp2[i,:])
        )
    end
    temp4
end
