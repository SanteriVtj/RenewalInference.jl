using RenewalInference
using Test, SafeTestsets

@safetestset "Tests for model struct" begin include("model_tests.jl") end
@safetestset "Tests for normal hazard" begin include("normhz_test.jl") end