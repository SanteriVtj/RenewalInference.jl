module RenewalInference

    using StatsBase, Distributions, 
        QuasiMonteCarlo, Random, Interpolations,
        LinearAlgebra, CairoMakie, DataFrames, KernelDensity,
        MLJ, SimpleChains, CSV, Distances, 
        ForwardDiff, StructArrays

    include("model_struct.jl")
    include("model.jl")
    include("threshold.jl")
    include("normalhz.jl")
    include("patenthz.jl")
    include("hz_functions.jl")
    include("util.jl")
    
    
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
        prepare_data,
        RRS,
        simulate,
        fval,
        MemAlloc,
        gen_sample
end
