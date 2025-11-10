# # Data loading
#
# ## Parsing the raw input
# The dataset we are working on is the collected works of Shakespeare. These texts are provided in the file
# `data/input.txt`. To work with this text data set, we have to load it into memory and identify the
# set of unique characters in the text that build up the vocabulary. Each character is assigned a unique token,
# a numerical representation of the character.
#
# The implementation below defines a struct to work with this text corpus.
# In the member variables we store the total length of the text, a block size that defines how many
# tokens are loaded in a batch, a dictionary that maps from individual characters to to tokens,
# another dictionary for the inverse mapping from tokens to characters, and a vector containing all tokens
# in the text.
#
# In the constructor method of the struct, we first read all lines and concatenate the characters into a single
# vector. Thenw e find all unique characters using the `unique` functions and sort them. We have to add the
# newline `\n` to the unique characters since it was stripped by the readlines function.
# Finally, we define dictionaries to map from character to token and the other way around.
#

struct NanoDataset
    length::Int64                   # Number of tokens in the text 
    block_size::Int64               # How long an observation is. This is important for the dataloader
    ch_to_int::Dict{Char, Int64}    # Maps chars to tokens
    int_to_ch::Dict{Int64, Char}    # Maps tokens to chars
    data::Vector{Int64}             # The tokens

    function NanoDataset(filename::String, block_size::Int64)
        lines = readlines(filename)             # Read all lines
        _out = [c for l ∈ lines for c ∈ l]      # Create char array
        chars = sort!(unique(_out))             # Sort all chars occuring in the dataset
        push!(chars, '\n')                      # Add the \n character that we stripped when only looking at lines
        ch_to_int = Dict(val => ix for (ix, val) in enumerate(chars))   # Mapping of chars to int
        int_to_ch = Dict(ix => val for (ix, val) in enumerate(chars))


        all_tokens = [ch_to_int[s] for s in join(lines, "\n")]

        new(length(all_tokens), block_size, ch_to_int, int_to_ch, all_tokens)
    end
end

# To create such a dataset we call the constructor
dataset = NanoDataset("../../data/input.txt", 16)

# We now define functions to encode strings into tokens and vice versa. These functions take
# the dataset as their first argument to make use of the dictionaries defined in the dataset
encode(d::NanoDataset, s::String) = [d.ch_to_int[ss] for ss in s]
decode(d::NanoDataset, i::AbstractArray{<:Integer}) = [d.int_to_ch[ii] for ii in i]

# Encoding a string to tokens is done like this:
encode(dataset, "Hello, World")

# We can also map from strings to token space and back
e = encode(dataset, "Hello, World")
decode(dataset, e)

# ## Data Loaders
# The Julian way of loading this text is by using a 
# [DataLoader](https://juliaml.github.io/MLUtils.jl/stable/api/#MLUtils.DataLoader) from 
# the  [MLUtils.jl](https://github.com/JuliaML/MLUtils.jl) package.
# These are analogous to f.ex. Pytorch Data Loaders.

using MLUtils

# It is pretty straightforward to define a DataLoader for our Dataset.
# The first thing we have to do is implement `numobs` and `getobs` methods for out dataset.
# These return the number of observations and a single observation respectively:

# For `numobs` it is enough to define `Base.length` for  NanoDataset:
Base.length(d::NanoDataset) = length(d.data) - d.block_size - 2

# Now we can do
numobs(dataset)


# For `getobs` it is enough to define `Base.getindex` for NanoDataset.
# What we want for auto-regressive text prediction is an (input,output) pair.
# For the dataset at hand, this translates into two sequences of length `block_size`, where
# the output is shifted one index ahead.
function Base.getindex(d::NanoDataset, i::Int)
    1 <= i <= d.length - d.block_size - 1|| throw(ArgumentError("Index is out of bounds"))
    return (d.data[i:i+d.block_size-1], d.data[i+1:i+d.block_size])
end

# Fetching the first observation returns the first (X,Y) training pair.
getobs(dataset, 1)

# To load minibatches, we need to define `getobs` for an array of indices like so:
MLUtils.getobs(d::NanoDataset, i::AbstractArray{<:Integer}) = [getobs(d, ii) for ii in i]

# If we now define a DataLoader, we can easily use it to fetch batches of shuffled observations:
dl = DataLoader(dataset, shuffle=true, batchsize=4, collate=true)
(x, y) = first(dl)

@show x

# The first batch of inputs is a Matrix of dimensions (block_size, batch_size). Julia stores
# matrices in column-major, so the first row with 16 tokens is stored consecutive in memory.
# The second row is the second sample and so on.


