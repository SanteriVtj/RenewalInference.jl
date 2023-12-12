function AEloss(par, md::ModelData, data::AEData; 
    μ=.5, σ=.005, chain=nothing, epoch=1, save=nothing
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
        TurboDense(tanh, 32),
        TurboDense(tanh, 32),
        TurboDense(tanh, 16),
        TurboDense(identity, 1)
    ) : chain

    X, Y = MLJ.partition((X,Y), 0.7, multi=true, shuffle=true)
    Xtrain, Xtest = X;
    Ytrain, Ytest = Y;
    pred = vec(NN_prediction(Xtrain', Ytrain', Xtest', Ytest', epoch=epoch, chain=chain))

    P = truncated(fit(Normal, pred), lower=0, upper=1)
    Q = truncated(Normal(μ,σ), lower=0, upper=1)
    
    err = kldivergence(P,Q)
    @info "KL-divergence, parameters:" err, par
    if ~isnothing(save)
        CSV.write(save, hcat(DataFrame(par', :auto), DataFrame("err"=>err)), append=isfile(save))
    end
    return err
end
