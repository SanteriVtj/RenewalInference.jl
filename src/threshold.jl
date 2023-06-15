function thresholds(par, c, z, o, increment=5)
    ϕ=par.ϕ;σⁱ=par.σⁱ;γ=par.γ;δ=par.δ;θ=par.θ;
    T = length(c)
    N = length(z)

    r_up = maximum(c)+increment
    r1 = 0:increment:r_up
    ngrid = length(r1)

    V = zeros(T, ngrid)
    r̄ = zeros(T)

    @inbounds V[T,:] = r1' .- c[T]
    idx = findfirst(V[T,:].>0)
    m_idx = maximum([idx-1,1])
    
    @inbounds r̄[T] = (r1[m_idx]*V[T,idx]-r1[idx]*V[T,m_idx])/
        (V[T,idx]-V[T,m_idx]);
    @inbounds V[T,:] = maximum(hcat(V[T,:], zeros(length(V[T,:]))), dims=2)

    μ, σ = log_norm_parametrisation(par, T)

    z = exp.(z.*σ'.+μ')
    o = o .≤ θ
    
    # @inbounds
    for t=T-1:-1:1
        temp1 = repeat(δ.*r1', N)
        temp2 = repeat(z[:,t],1,ngrid)
        temp3 = repeat(s[:,t],1,ngrid)
        interp = linear_interpolation(r1, V(:,t))
        temp4 = interp(temp3.*maximum([temp2;;;temp3], dims=3))
    end
    [116, 138, 169, 201, 244, 286, 328, 381, 445, 508, 572, 646, 720, 794, 868, 932, 995]
end
