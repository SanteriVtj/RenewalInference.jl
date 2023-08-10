module RenewalInference

    using StatsBase, Distributions, 
        QuasiMonteCarlo, Random, Interpolations, DualNumbers,
        LinearAlgebra

    include("model.jl")
    include("threshold.jl")
    include("normalhz.jl")
    include("patenthz.jl")
    include("hz_functions.jl")
    
    
    export NormModel,
        simulate_normhz,
        normhz,
        PatentModel,
        simulate_patenthz,
        patenthz
end
