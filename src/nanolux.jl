module NanoLux

export DATAFILE
const DATAFILE = "data/input.txt"

using Distributions
using MLUtils
using NNlib
using Random


include("dataloader.jl")
include("utils.jl")

end # module nanolux
