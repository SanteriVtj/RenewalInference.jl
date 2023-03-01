module RenewalInference

    using StatsBase, StaticArrays

    include("model.jl")
    include("single_country.jl")

    export RenewalModel, 
        single_country!

end
