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
    r::Matrix{T}
    r_d::BitMatrix
    ν::T
    β::T
    nt::Int
    ngrid::Int
    V::Matrix{T}
    controller::ModelControl
    function ModelData{T}(
        hz, costs, X, x, obsolence, r, r_d, ngrid, V,
        nt, controller;
        ν=2., β=.95
    ) where {T<:AbstractFloat}
        # Define dimensions
        t = length(hz)
        N = size(X, 1)
        
        # Test that each variable has correct time dimension
        t == size(x, 2) ? nothing : throw(AssertionError("Length of hazards and simulation periods doesn't match."))
        t == length(costs) ? nothing : throw(AssertionError("Length of hazards and costs doesn't match."))
        t == size(obsolence, 2) + 1 ? nothing : throw(AssertionError("Length of hazards and obsolences doesn't match."))
        t == size(r, 2) ? nothing : throw(AssertionError("Length of hazards and simulation time doesn't match."))
        t == size(r_d, 2) ? nothing : throw(AssertionError("Length of hazards and simulation dummy time doesn't match."))
        t == size(V, 1) ? nothing : throw(AssertionError("Length of hazards and value discretised value functions time dimension doesn't match."))
        
        # Test that each variable has correct sample size
        N == size(obsolence, 1) ? nothing : throw(AssertionError("Number of observations and simulation obsolences doesn't match."))
        N == size(x, 1) ? nothing : throw(AssertionError("Number of observations and simulation value sample doesn't match."))
        N == size(r, 1) ? nothing : throw(AssertionError("Number of observations and preallocation for values doesn't match."))
        N == size(r_d, 1) ? nothing : throw(AssertionError("Number of observations and simulation dummies doesn't match."))
        
        # If all tests are satisfied, create new instance
        new(hz, costs, X, x, obsolence, r, r_d, ν, β, nt, ngrid, V, controller)
    end
end

# Function for generating new instance of ModelData with default (Halton)samples
# and other preallocated matrices.
function ModelData(hz::Vector{T}, costs::Vector{T}, X::Matrix{T};
    alg=QuasiMonteCarlo.HaltonSample(), ngrid=500,
    nt=Threads.nthreads(), controller=ModelControl()
) where {T<:AbstractFloat}
    # Inference dimensions for simulation draws
    N = size(X, 1)
    t = length(hz)

    # Generate quasi monte carlo draws
    obsolence = Matrix(QuasiMonteCarlo.sample(N,t-1,alg)')
    x = Matrix(QuasiMonteCarlo.sample(N,t,alg)')
    # Preallocate memory for value function matrices
    r = zeros(T,N,t)
    r_d = falses(N, t)

    # Discretisation for value function
    V = Matrix(undef, t, ngrid)

    return ModelData{T}(
        hz, costs, X, x, obsolence, r, r_d, ngrid, V, nt, controller
    )
end