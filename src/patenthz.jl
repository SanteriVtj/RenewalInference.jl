function patenthz(par::PatentModel, x)
    ϕ=par.ϕ;σⁱ=par.σⁱ;γ=par.γ;δ=par.δ;θ=par.θ;
end

function simulate_patenthz(par::PatentModel, x, s)
    """
    simulate_patenthz(par::PatentModel, x)

    Function for simulating hazard rates for patent expirations.

        # Arguments:
        par::PatentModel: struct containing parameters for the distribution of patent exirations.
        x: random matrix distributed as U([0,1]) for drawing the values from LogNormal.
        s: obsolence draw. Also U([0,1]) distributed random matrix.
    """
    ϕ=par.ϕ;σⁱ=par.σⁱ;γ=par.γ;δ=par.δ;θ=par.θ;
    threshold = collect(35:5:80)

    n, T = size(x)
    q = @view x[:,1]
    z = @view x[:,2:end]

    # Conversion of mean and variance for log normal distribution according to the normal specification
    e_mean = ϕ.^(1:T)*σⁱ*(1-γ)
    e_var = e_mean.^2

    μ = 2*log.(e_mean)-1/2*log.(e_mean.^2+e_var)
    σ = sqrt.(-2*log.(e_mean)+log.(e_var+e_mean.^2))

    # Zero matrix for patent values and active patent periods
    r = zeros(n,T)
    r_d = zeros(n,T)
    obsolence = zeros(n,T-1)

    r[:,1] .= quantile.(LogNormal(μ[1], σ[1]), q)
    r_d[:,1] .= r[:,1] .≥ threshold[1]

    # size(z)=n×T⇒size(g(z))=T×n⇒size(g(z))'=n×T i.e. size(z)=n×T before and after this line (at least that is the intent)
    learning = quantile.(LogNormal.(μ[2:end], σ[2:end]), z')'
    obsolence .= s .≤ θ

    # Computing patent values for t=2,…,T
    @inbounds for t=2:T
        # compute patent value at t by maximizing between learning shocks and depreciation
        r[:,t] .= s[:,t-1].*maximum(hcat(δ.*r[:,t-1], learning[:,t-1]), dims=2) # concat as n×2 matrix and choose maximum in for each row
        # If patent wasn't active in t-1 it cannot be active in t
        r[r_d[:,t-1] .== 0,t] .= 0
        # Patent is kept active if its value exceed the threshold o.w. set to zero
        r_d[:,t] .= r[:,t] .≥ threshold[t]
    end

    (r, r_d)
end