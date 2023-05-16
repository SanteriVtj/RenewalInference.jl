using RenewalInference
using Test, SafeTestsets

@safetestset "Tests for model struct" begin include("model_tests.jl") end
