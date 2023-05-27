function patenthz(x, par::PatentModel)
    ϕ=par.ϕ;σⁱ=par.σⁱ;γ=par.γ;δ=par.δ;θ=par.θ;T=par.T;n=par.n;
end

function simulate_patenthz(par::PatentModel, z)
    ϕ=par.ϕ;σⁱ=par.σⁱ;γ=par.γ;δ=par.δ;θ=par.θ;T=par.T;n=par.n;

    e_mean = ϕ.^(1:T)*σⁱ*(1-γ)
    e_var = e_mean.^2

    μ = 2*log.(e_mean)-1/2*log.(e_mean.^2+e_var)
    σ = sqrt.(-2*log.(e_mean)+log.(e_var+e_mean.^2))

    r = zeros(n,T)

    # initial_value = quantile(LogNormal(μ[1], σ[1]), q)

    # obsolence = quantile.(LogNormal.(μ[2:end], σ[2:end]), z')' # size(z)=n×T⇒size(g(z))=T×n⇒size(g(z))'=n×T i.e. size(z)=n×T before and after this line (at least that is the intent)
end