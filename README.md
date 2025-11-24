# NanoGPT in Julia's Lux
This repository re-implements [Andrej Karpathy's NanoGPT](https://github.com/karpathy/nanoGPT) using Julia's 
[Lux](https://lux.csail.mit.edu/).

There code does essentially the same as in the Videos, but it's re-styled to use Lux.
Significant differences include
1. Making use of the [Training API](https://lux.csail.mit.edu/stable/api/Lux/utilities#Training-API)
2. The use of column-major ordering in Julia vs row-major in python.

## Installation
To run this package, install Julia using [juliaup](https://julialang.org/install/).
As of November 2025, please run with Julia v1.11 instead of v1.12. The port of Enzyme and Reactant to Julia 1.12
is [on the way](https://github.com/EnzymeAD/Enzyme.jl/issues/2699).

Then, install the necessary requirements
```
(NanoLux) pkg> instantiate
(NanoLux) pkg> update
```

## How to run
Training loops are implemented for both, `reactant` and `zygote`, backends. Run either one through

```julia
julia --project=. src/runme_enzyme.jl
julia --project=. src/runme_zygote.jl
```

Command line arguments are
* `--n_embd=64`: Number of embedding dimensions
* `--batch_size=32`: Batch size
* `--block_size=64`: Block size (the sequence length)
* `--head_size=16`: Length of transformer head size
* `--num_iter=5_000`: Number of batches over which to train


## Benchmarks
On my machine (Mac M4 Pro) I get the following performance for
`n_embd=64` `batch_size=32` `block_size=64` `head_size=16` `num_iter=5_000`


| Metric                  | Reactant  | Zygote      |
|-------------------------|-----------|-------------|
| Train 1st batch:        | 120.12s   |
| Avg batch training time | ~27s      |
| Total training time     | ~536s     |
| Final loss (train)      | 1.54838   |
| Final loss (valid)      | 1.78198   |


## Output

An output sample after running with the default parameters:
```
ee
As mountain winds: but then exactly do
All points of my commalter, sishiredly, prove
Let speak here obe your eaple of thee:
As maree, leeds ias he pitter hegge we instrued,
Let, care depurn, with scrow; all lethy put friends
The officitre one this screetnexding%
```
