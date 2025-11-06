using Test
using MLUtils
using Lux
using Random
using WeightInitializers

using NanoLux

# List of test files to run 


all_test_names = [
    "dataloaders",
]

# If arguments are passed, parse the test names to run from the arguments.
# Otherwise run all tests
const tests = !(isempty(ARGS)) ? ARGS : all_test_names

# Run the tests 
@testset "All tests" begin
    for test ∈ tests
        include("test_" * test * ".jl")
    end
end
