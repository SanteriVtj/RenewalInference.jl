function single_country!(model::RenewalModel, param::V, error_function=msd) where {V<:AbstractVector}
    """
    single_country!

    With this function one can step single country renewal model forwards.
    By passing this to optimizer one can 
    ...
    # Arguments
    - `model::RenewalModel`: data struct for necessary data.
    - `param::V`: parameter vector.
    - `error_function`: preferred error function.
    ...
    """
    
    δ = param[1]
    θ = param[2]
    ϕ = param[3]
    σ = param[4]
    γ = param[5]

    expiration = model.expiration
    r̄ = model.fees
    T = model.max_time
    draw = model.draw

    
    @SVector error_function(expiration, hz_model)
end

