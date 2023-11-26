mutable struct ModelControl
    simulation::Bool
    x_transformed::Bool
    debug::Bool
    ModelControl(simulation=false, x_transformed=false, debug=false) = new(simulation, x_transformed, debug)
end

struct ModelData{T<:AbstractFloat}
    hz::Vector{T}
    costs::Vector{T}
    X::Matrix{T}
    x::Matrix{T}
    obsolence::Matrix{T}
    ν::T
    β::T
    ngrid::Int
    controller::ModelControl
    function ModelData{T}(
        hz, costs, X, x, obsolence, ngrid, controller;
        ν=2., β=.95
    ) where {T<:AbstractFloat}
        # Define dimensions
        t = length(hz)
        N = size(X, 1)
        
        # Test that each variable has correct time dimension
        t == size(x, 2) ? nothing : throw(AssertionError("Length of hazards and simulation periods doesn't match."))
        t == length(costs) ? nothing : throw(AssertionError("Length of hazards and costs doesn't match."))
        t == size(obsolence, 2) + 1 ? nothing : throw(AssertionError("Length of hazards and obsolences doesn't match."))
        
        # Test that each variable has correct sample size
        N == size(obsolence, 1) ? nothing : throw(AssertionError("Number of observations and simulation obsolences doesn't match."))
        N == size(x, 1) ? nothing : throw(AssertionError("Number of observations and simulation value sample doesn't match."))
        
        # If all tests are satisfied, create new instance
        new(hz, costs, X, x, obsolence, ν, β, ngrid, controller)
    end
end

# Function for generating new instance of ModelData with default (Halton)samples
# and other preallocated matrices.
function ModelData(hz::Vector{T}, costs::Vector{T}, X::Matrix{T};
    alg=QuasiMonteCarlo.HaltonSample(), ngrid=500, controller=ModelControl()
) where {T<:AbstractFloat}
    # Inference dimensions for simulation draws
    N = size(X, 1)
    t = length(hz)

    # Generate quasi monte carlo draws
    obsolence = Matrix(QuasiMonteCarlo.sample(N,t-1,alg)')
    x = Matrix(QuasiMonteCarlo.sample(N,t,alg)')

    return ModelData{T}(
        hz, costs, X, x, obsolence, ngrid, controller
    )
end