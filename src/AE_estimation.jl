function AEloss(par, md::ModelData, data::AEData; 
    μ=.5, σ=.005, chain=nothing, epoch=1, save=nothing, save_pred=nothing
)
    stopping = patenthz(par, md)
    Xₘ = prepare_data(md)
    Xₘ = hcat(Xₘ, stopping)
    Xₑ = data.Xₑ
    Yₘ = data.Yₘ
    Yₑ = data.Yₑ

    X = vcat(Xₘ, Xₑ)
    Y = vcat(Yₘ, Yₑ)
    N, K = size(X)

    chain = isnothing(chain) ? SimpleChain(
        static(K),
        TurboDense(tanh, 16),
        TurboDense(tanh, 16),
        TurboDense(tanh, 8),
        TurboDense(x->min(max(0, x), 1), 1)
    ) : chain

    X, Y = MLJ.partition((X,Y), 0.7, multi=true, shuffle=true)
    Xtrain, Xtest = X;
    Ytrain, Ytest = Y;
    pred = vec(NN_prediction(Xtrain', Ytrain', Xtest', Ytest', epoch=epoch, chain=chain))

    # P = fit(Normal, pred)
    # Q = Normal(μ,σ)

    # Discretize Normal with very small variance
    discretization_range = LinRange(0,1,Int(floor(sqrt(length(pred)))))
    P = normalize(fit(Histogram, pred, discretization_range), mode=:probability).weights
    F(x) = cdf(truncated(Normal(.5,.009), lower=0, upper=1), x)
    Q = [F(discretization_range[i])-F(discretization_range[i-1]) for i in 2:length(discretization_range)]
    # # Fill probability zero events in support with small probability
    P[P.≈0] .= 1e-8
    Q[Q.≈0] .= 1e-8
    # Normalize
    P = P./sum(P)
    Q = Q./sum(Q)
    
    err = kl_divergence(P,Q)
    @info "KL-divergence, parameters:" err, par
    if ~isnothing(save)
        CSV.write(save, hcat(DataFrame(par', :auto), DataFrame("err"=>err)), append=isfile(save))
    end
    if ~isnothing(save_pred)
        CSV.write(save_pred, DataFrame(pred', :auto), append=isfile(save_pred))
    end
    return err
end
