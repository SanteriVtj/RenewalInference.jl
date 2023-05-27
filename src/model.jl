struct NormModel{T<:Real}
    μ::Real
    σ::Real
    n::Int64
    c::Vector{T}
    ν::Float64
end

struct PatentModel
    ϕ::Float64
    σⁱ::Float64
    γ::Float64
    δ::Float64
    θ::Float64
    T::Int64
end
