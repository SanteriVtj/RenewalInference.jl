module RenewalInference

    using StatsBase, StaticArrays, Distributions, 
        QuasiMonteCarlo, Random

    include("model.jl")
    include("normalhz.jl")
    include("patenthz.jl")
    include("hz_functions.jl")
    if pwd()=="C:\\Users\\Santeri\\.julia\\dev\\RenewalInference" include("runmodel.jl") end

    if pwd()=="C:\\Users\\Santeri\\.julia\\dev\\RenewalInference"
        export NormModel,
        simulate_normhz,
        normhz,
        runnorm,
        PatentModel,
        simulate_patenthz,
        patenthz,
        runpatent,
        runpm,
        computehz
    else    
        export NormModel,
            simulate_normhz,
            normhz,
            PatentModel,
            simulate_pathenthz,
            patenthz
    end
end
