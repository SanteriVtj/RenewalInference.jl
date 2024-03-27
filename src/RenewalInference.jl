module RenewalInference

    using StatsBase, Distributions, 
        QuasiMonteCarlo, Random, Interpolations,
        LinearAlgebra, CairoMakie, DataFrames, KernelDensity,
        MLJ, AdversialEstimation, SimpleChains, CSV, Distances, ForwardDiff

    include("model_struct.jl")
    include("model.jl")
    include("threshold.jl")
    include("normalhz.jl")
    include("patenthz.jl")
    include("hz_functions.jl")
    include("util.jl")
    include("AE_estimation.jl")
    
    
    export NormModel,
        simulate_normhz,
        normhz,
        PatentModel,
        simulate_patenthz,
        patenthz,
        patentSMM,
        ModelData,
        ModelControl,
        plot_paramdist,
        AEestimation,
        AEData,
        AEloss,
        prepare_data
end
