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
    function ModelData{T}(
        hz, costs, X, x, obsolence, r, r_d; 
        ν=2., β=.95, nt=Threads.nthreads()) where {T<:AbstractFloat}
        # Define dimensions
        t = length(hz)
        N = size(X, 1)
        
        # Test that each variable has correct time dimension
        t == size(x, 2) ? nothing : throw(AssertionError("Length of hazards and simulation periods doesn't match."))
        t == length(costs) ? nothing : throw(AssertionError("Length of hazards and costs doesn't match."))
        t == size(obsolence, 2) + 1 ? nothing : throw(AssertionError("Length of hazards and obsolences doesn't match."))
        t == size(r, 2) ? nothing : throw(AssertionError("Length of hazards and simulation time doesn't match."))
        t == size(r_d, 2) ? nothing : throw(AssertionError("Length of hazards and simulation dummy time doesn't match."))

        # Test that each variable has correct sample size
        N == size(obsolence, 1) ? nothing : throw(AssertionError("Number of observations and simulation obsolences doesn't match."))
        N == size(x, 1) ? nothing : throw(AssertionError("Number of observations and simulation value sample doesn't match."))
        N == size(r, 1) ? nothing : throw(AssertionError("Number of observations and preallocation for values doesn't match."))
        N == size(r_d, 1) ? nothing : throw(AssertionError("Number of observations and simulation dummies doesn't match."))
        
        # If all tests are satisfied, create new instance
        new(hz, costs, X, x, obsolence, r, r_d, ν, β, nt)
    end
end

function ModelData(hz::Vector{T}, costs::Vector{T}, X::Matrix{T};
    alg=QuasiMonteCarlo.HaltonSample()) where {T<:AbstractFloat}
    # Inference dimensions for simulation draws
    N = size(X, 1)
    t = length(hz)

    # Generate quasi monte carlo draws
    obsolence = Matrix(QuasiMonteCarlo.sample(N,t-1,alg)')
    x = Matrix(QuasiMonteCarlo.sample(N,t,alg)')
    # Preallocate memory for value function matrices
    r = zeros(T,N,t)
    r_d = falses(N, t)

    return ModelData{T}(
        hz, costs, X, x, obsolence, r, r_d
    )
end