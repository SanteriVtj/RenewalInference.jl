function single_country!(model::RenewalModel, error_function=msd)
    """
    single_country!

    With this function one can step single country renewal model forwards.
    By passing this to optimizer one can 
    ...
    # Arguments
    - `model::RenewalModel`: data and parameters.
    - `error_function`: preferred error function.
    ...
    """
    δ = model.param[1]
    θ = model.param[2]
    ϕ = model.param[3]
    σ = model.param[4]
    γ = model.param[5]

    expiration = model.expiration
    r̄ = model.fees
    T = model.T
    P = model.max_time

    
    error_function(expiration, hz_model)
end

