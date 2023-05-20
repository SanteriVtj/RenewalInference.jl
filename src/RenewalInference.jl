module RenewalInference

    using StatsBase, StaticArrays, Distributions, 
        QuasiMonteCarlo, Random

    include("model.jl")
    include("normalhz.jl")

    export NormModel,
        simulate_normhz,
        normhz

end
