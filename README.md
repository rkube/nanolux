# NanoGPT in Julia's Lux
This repository re-implements [Andrej Karpathy's NanoGPT](https://github.com/karpathy/nanoGPT) using Julia's 
[Lux](https://lux.csail.mit.edu/).

There code does essentially the same as in the Videos, but has been re-styled to in the appropriate way for Lux.
Significant differences include
1. Making use of the [Training API](https://lux.csail.mit.edu/stable/api/Lux/utilities#Training-API)
2. The use of column-major ordering in Julia vs row-major in python.

## Installation
To run this package, install Julia using [juliaup](https://julialang.org/install/).
As of November 2025, please install Julia v1.11 instead of v1.12. The port of Enzyme and Reactant to Julia 1.12
has not been completed.

Then, install the necessary requirements
```
(NanoLux) pkg> instantiate
(NanoLux) pkg> update
```

## How to run

The scripts `training...v1.jl` train a simple model that only includes learnable embedding layers.
These can be run as
```julia
julia --project=. src/training_enzyme_clean_v1.jl
```

and similar for Zygote.


