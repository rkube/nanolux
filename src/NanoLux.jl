module NanoLux

export DATAFILE
const DATAFILE = "data/input.txt"

using Distributions
using LinearAlgebra
using Lux
using MLUtils
using NNlib
using Random


include("dataloader.jl")
include("utils.jl")
include("layers.jl")

end # module nanolux
