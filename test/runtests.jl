using RenewalInference
using Test, SafeTestsets

@safetestset "Tests for model struct" begin include("model_tests.jl") end
@safetestset "Tests for single country inference" begin include("single_country_tests.jl") end
