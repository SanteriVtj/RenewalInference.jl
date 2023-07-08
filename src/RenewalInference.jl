module RenewalInference

    using StatsBase, StaticArrays, Distributions, 
        QuasiMonteCarlo, Random, Interpolations, DualNumbers

    include("model.jl")
    include("threshold.jl")
    include("normalhz.jl")
    include("patenthz.jl")
    include("hz_functions.jl")
    
    if pwd()=="C:\\Users\\Santeri\\.julia\\dev\\RenewalInference" 
        
        include("runmodel.jl") 
        
        export NormModel,
        simulate_normhz,
        normhz,
        runnorm,
        PatentModel,
        simulate_patenthz,
        patenthz,
        runpatent,
        runpm,
        computehz,
        thresholds,
        log_norm_parametrisation
    else    
        export NormModel,
            simulate_normhz,
            normhz,
            PatentModel,
            simulate_pathenthz,
            patenthz
    end
end
