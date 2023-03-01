struct RenewalModel{T<:Number}
    expiration::Vector{T}
    max_time::T
    N::Integer
    param::Vector{Float64}
end

