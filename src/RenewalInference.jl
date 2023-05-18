module RenewalInference

    using StatsBase, StaticArrays

    include("model.jl")
    include("normalhz.jl")

    export RenewalModel,
        simulate_normhz,
        normhz

end
