mutable struct ModelControl
    simulation::Bool
    debug::Bool
    ae_mode::Bool
    ModelControl(;simulation=false, debug=false, ae_mode=false) = new(simulation, debug, ae_mode)
end

struct ModelData
    hz::Vector{Float64}
    costs::Vector{Float64}
    X::Matrix{Float64}
    s_data::Matrix{Float64}
    x::Matrix{Float64}
    obsolence::Matrix{Float64}
    β::Float64
    renewals::Vector{Float64}
    ngrid::Int16
    controller::ModelControl
    alg::Union{Sampleable, SamplingAlgorithm}
    function ModelData(
        hz, costs, X, s_data, x, obsolence, renewals, ngrid, controller;
        β=.95, alg=QuasiMonteCarlo.HaltonSample()
    )
        # Define dimensions
        t = length(hz)
        N = size(X, 1)
        
        # Test that each variable has correct time dimension
        t == length(costs) ? nothing : throw(AssertionError("Length of hazards and costs doesn't match."))
        
        # Test that each variable has correct sample size
        N == size(s_data, 1) ? nothing : throw(AssertionError("Number of observations and data for σⁱ doesn't match."))
        
        # If all tests are satisfied, create new instance
        new(hz, costs, X, s_data, x, obsolence, β, renewals, ngrid, controller, alg)
    end
end

# Function for generating new instance of ModelData with default (Halton)samples
# and other preallocated matrices.
function ModelData(hz::Vector{Float64}, costs::Vector{Float64}, X::Matrix{Float64}, s_data::Matrix{Float64}, renewals::Vector{Float64};
    alg=QuasiMonteCarlo.HaltonSample(), ngrid=1000, controller=ModelControl(), β=.95
)
    # Inference dimensions for simulation draws
    N = size(X, 1)
    t = length(hz)

    # Generate quasi monte carlo draws
    obsolence = QuasiMonteCarlo.sample(t-1,1,alg)
    x = QuasiMonteCarlo.sample(t,1,alg)

    return ModelData(
        hz, costs, X, s_data, x, obsolence, renewals, ngrid, controller, alg=alg, β=β
    )
end

# Helper functions and data structure for adversial estimation
function prepare_data(md::ModelData)
    s_data = md.s_data
    X = md.X

    x = hcat(s_data, X)

    return x
end

struct AEData
    Xₑ::Matrix{Float64}
    Yₑ::Matrix{Float64}
    Yₘ::Matrix{Float64}
    function AEData(Xₑ, Yₑ, Yₘ)
        size(Xₑ, 1) == size(Yₑ, 1) ? nothing : throw(AssertionError("Size of Xₑ and Yₑ doesn't match."))
        size(Xₑ, 1) == size(Yₘ, 1) ? nothing : throw(AssertionError("Size of Xₑ and Yₑ doesn't match."))
        
        new(Xₑ, Yₑ, Yₘ)
    end
end

function AEData(Xₑ::Matrix{Float64})
    N, K = size(Xₑ)

    Yₑ = ones(T, N, 1)
    Yₘ = zeros(T, N, 1)

    return AEData(Xₑ, Yₑ, Yₘ)
end

# Return struct
struct RRS
    r_d
    r
end