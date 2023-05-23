module RenewalInference

    using StatsBase, StaticArrays, Distributions, 
        QuasiMonteCarlo, Random

    include("model.jl")
    include("normalhz.jl")
    if pwd()=="C:\\Users\\Santeri\\.julia\\dev\\RenewalInference" include("runmodel.jl") end

    if pwd()=="C:\\Users\\Santeri\\.julia\\dev\\RenewalInference"
        export NormModel,
        simulate_normhz,
        normhz,
        runnorm
    else    
        export NormModel,
            simulate_normhz,
            normhz
    end
end
