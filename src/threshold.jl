function thresholds(par, modeldata)
    ϕ, σⁱ, γ, δ, θ = par
    c = modeldata.costs
    x = modeldata.x
    obsolence = modeldata.obsolence
    X = modeldata.X
    β = modeldata.β
    ngrid = modeldata.ngrid
    
    
    T = length(c)
    N, M = size(x)
    
    V = zeros(eltype(par), T, ngrid)
    r1 = collect(range(0, maximum(c), length=ngrid-1))
    append!(r1, last(r1)+last(diff(r1)))
   
    r̄ = zeros(eltype(par), T)

    # # Compute values for t=T i.e. the last period from which the backwards induction begins
    @inbounds begin 
        V[T,:] = r1' .- c[T]
        idx = IfElse.ifelse(
            any(V[T,:].>zero(eltype(V))), 
            findfirst(V[T,:].>zero(eltype(V))), 
            1
        )
        m_idx = maximum([idx-1,1])

        r̄[T] = (r1[m_idx]*V[T,idx]-r1[idx]*V[T,m_idx])/
            (V[T,idx]-V[T,m_idx]);
        V[T,:] = maximum(hcat(V[T,:], zeros(length(V[T,:]))), dims=2)
    end

    # μ, σ = log_norm_parametrisation(par, T)
    μ, σ = initial_shock_parametrisation(par, X)

    if ~modeldata.controller.x_transformed
        modeldata.controller.x_transformed = true
        x[:,1] .= quantile.(LogNormal.(μ, σ), x[:,1])
        x[:,2:end] .= -(log.(1 .-x[:,2:end]).*ϕ.^(1:T-1)'.*σⁱ.-γ)
    end
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


