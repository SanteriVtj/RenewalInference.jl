function thresholds(par, modeldata)
    ϕ, γ, δ, θ = par
    c = modeldata.costs
    x = modeldata.x
    obsolence = modeldata.obsolence
    X = modeldata.X
    β = modeldata.β
    ngrid = modeldata.ngrid
    s_data = modeldata.s_data
    
    
    T = length(c)
    N, M = size(x)
    σⁱ = hcat(ones(N), s_data)*par[6+size(X,2)+1:6+size(X,2)+1+size(s_data, 2)]
    V = zeros(eltype(par), T, ngrid)
    # r1 = exp.(LinRange(log(.00001), log(maximum(c)+maximum(c)/ngrid), ngrid))
    r1 = LinRange(0, maximum(c)+maximum(c)/ngrid, ngrid)
    
    r̄ = zeros(eltype(par), T)

    # # Compute values for t=T i.e. the last period from which the backwards induction begins
    @inbounds begin 
        V[T,:] = r1' .- c[T]
        idx = findfirst(V[T,:].>zero(eltype(V)))
        idx = isnothing(idx) ? 1 : idx
        m_idx = max(idx-1,1)

        r̄[T] = (r1[m_idx]*V[T,idx]-r1[idx]*V[T,m_idx])/
            (V[T,idx]-V[T,m_idx]);
        V[T,:] = max(V[T,:], zeros(length(V[T,:])))
    end

    # μ, σ = log_norm_parametrisation(par, T)
    μ, σ = initial_shock_parametrisation(par, X)

    repopulate_x!(modeldata)
    x[:,1] .= quantile.(LogNormal.(μ, σ), x[:,1])
    x[:,2:end] .= -(log.(1 .-x[:,2:end]).*ϕ.^(1:T-1)'.*σⁱ.-γ)
    o = obsolence .≤ θ
    
    temp4 = zeros(eltype(V), N, ngrid)
    @inbounds for t=T-1:-1:1
        # Allocation for temp variables
        temp1 = δ.*r1
        temp2 = @view x[:,t]
        temp3 = @view o[:,t]
        interp = @views linear_interpolation(
            r1,
            V[t+1, :], 
            extrapolation_bc=Line()
        )
        _calctemp4!(temp4, temp1,temp2,temp3,interp)
        temp5 = mean(temp4, dims=1)
        # Compute patent values
        V[t,:] = r1'.-c[t].+β.*temp5

        # Gather positive values
        idx = findfirst(V[t,:].>zero(eltype(V)))
        r̄[t] = (idx == 1) | isnothing(idx) ? 0. : (r1[idx-1]*V[t,idx]-r1[idx]*V[t,idx-1])/(V[t,idx]-V[t,idx-1])
        V[t,:] = maximum([V[t,:] zeros(size(V[t,:]))], dims=2)
    end
    return r̄
end

function _calctemp4!(temp4, temp1,temp2,temp3,interp)
    Threads.@threads for i in 1:length(eachrow(temp4))
        temp4[i,:] .= @views interp.(
            temp3[i].*max.(temp1, temp2[i])
        )
    end
end


