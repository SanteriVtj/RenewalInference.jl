mutable struct ModelControl
    simulation::Bool
    debug::Bool
    ae_mode::Bool
    ModelControl(;simulation=false, debug=false, ae_mode=false) = new(simulation, debug, ae_mode)
end

struct ModelData{T<:Real}
    hz::Vector{T}
    costs::Vector{T}
    X::Matrix{T}
    s_data::Matrix{T}
    x::Matrix{T}
    obsolence::Matrix{T}
    β::T
    renewals::Vector{T}
    ngrid::Int
    controller::ModelControl
    alg::Union{Sampleable, SamplingAlgorithm}
    nt::Int
    function ModelData{T}(
        hz, costs, X, s_data, x, obsolence, renewals, ngrid, controller, nt;
        β=.95, alg=QuasiMonteCarlo.HaltonSample()
    ) where {T<:Real}
        # Define dimensions
        t = length(hz)
        N = size(X, 1)
        
        # Test that each variable has correct time dimension
        t == length(costs) ? nothing : throw(AssertionError("Length of hazards and costs doesn't match."))
        
        # Test that each variable has correct sample size
        N == size(s_data, 1) ? nothing : throw(AssertionError("Number of observations and data for σⁱ doesn't match."))
        
        # If all tests are satisfied, create new instance
        new(hz, costs, X, s_data, x, obsolence, β, renewals, ngrid, controller, alg, nt)
    end
end

# Function for generating new instance of ModelData with default (Halton)samples
# and other preallocated matrices.
function ModelData(hz::Vector{T}, costs::Vector{T}, X::Matrix{T}, s_data::Matrix{T}, renewals::Vector{T};
    alg=QuasiMonteCarlo.HaltonSample(), ngrid=1000, controller=ModelControl(), nt=Threads.nthreads(),
    β=.95, S=1000
) where {T<:Real}
    # Inference dimensions for simulation draws
    N = size(X, 1)
    t = length(hz)

    # Generate quasi monte carlo draws
    obsolence = QuasiMonteCarlo.sample(S,1,alg)
    x = QuasiMonteCarlo.sample(S,1,alg)

    return ModelData{T}(
        hz, costs, X, s_data, x, obsolence, renewals, ngrid, controller, nt, alg=alg, β=β
    )
end

function repopulate_x!(md::ModelData)
    N,K = size(md.x)
    @views md.x[:,:] = Matrix(QuasiMonteCarlo.sample(N,K,md.alg)')
end

function prepare_data(md::ModelData)
    s_data = md.s_data
    X = md.X

    x = hcat(s_data, X)

    return x
end

struct AEData{T<:AbstractFloat}
    Xₑ::Matrix{T}
    Yₑ::Matrix{T}
    Yₘ::Matrix{T}
    function AEData{T}(Xₑ, Yₑ, Yₘ) where {T<:AbstractFloat}
        size(Xₑ, 1) == size(Yₑ, 1) ? nothing : throw(AssertionError("Size of Xₑ and Yₑ doesn't match."))
        size(Xₑ, 1) == size(Yₘ, 1) ? nothing : throw(AssertionError("Size of Xₑ and Yₑ doesn't match."))
        
        new(Xₑ, Yₑ, Yₘ)
    end
end

function AEData(Xₑ::Matrix{T}) where {T<:AbstractFloat}
    N, K = size(Xₑ)

    Yₑ = ones(T, N, 1)
    Yₘ = zeros(T, N, 1)

    return AEData{T}(Xₑ, Yₑ, Yₘ)
end
