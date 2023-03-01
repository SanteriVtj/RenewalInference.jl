struct RenewalModel{T<:Number}
    expiration::Vector{T}
    fees::Vector{Float64}
    max_time::T
    N::Integer
    param::Vector{Float64}

    function RenewalModel{T}(expiration, fees, max_time, N, param) where T<:Number
        if max_time != first(size(fees))
            error("Fees need to be defined for each period")
        else
            new(expiration, fees, max_time, N, param)
        end
    end
end

