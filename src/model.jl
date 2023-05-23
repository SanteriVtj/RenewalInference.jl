struct NormModel{T<:Real}
    μ::Real
    σ::Real
    n::Int64
    c::Vector{T}
    ν::Real
end
