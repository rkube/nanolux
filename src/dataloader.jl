
using OneHotArrays
using MLUtils
using Random

function load_dataset(filename)

    lines = readlines(filename)

    _out = [c for l ∈ lines for c ∈ l]
    chars = sort!(unique(_out))
    push!(chars, '\n')
    vocab_size = length(chars)
    int_to_ch = Dict( ix => val for (ix, val) in enumerate(chars) )
    ch_to_int = Dict( val => ix for (ix, val) in enumerate(chars) )
    encode(str) = [ch_to_int[s] for s in str]
    decode(tok) = [int_to_ch[t] for t in tok]

    data = encode(join(lines, "\n"))
    return data, encode, decode, vocab_size
end


"""
    get_batch(dataset, batch_size, block_size)

Return batches of data from a dataset

# Arguments
- `dataset` - The dataset, either data_train or data_test
- `batch_size` - Number of independent sequences 
- `block_size` - Length of sequences
# Returns:
- `x` - 
"""
function get_batch(rng, dataset, batch_size, block_size)
    N_data = length(dataset) - block_size
    ix = rand(rng, 1:N_data, batch_size) # Generate batch_size number of random offsets
    x = stack([dataset[i:i+block_size-1] for i in ix], dims=2)   # Stack sequences with random offsets into a matrix
    y = stack([dataset[i+1:i+block_size] for i in ix], dims=2)

    return x, y
end


rng = Random.default_rng()
Random.seed!(rng, 1337)
batch_size = 4
block_size = 8

dataset, encode_fun, decode_fun, vocab_size = load_dataset("data/input.txt")
N = length(dataset)
N_train = Int(round(0.9 * N))
data_train = dataset[1:N_train]
data_test = dataset[N_train+1:end]

# Test the model on reactant arrays
xb, yb = get_batch(rng, dataset, batch_size, block_size)
# Make yb a one-hot batch
yb_hot = onehotbatch(yb, 1:65)



export NanoDataset, encode, decode

"""
    NanoDataset

This struct defines the text data set we operate on.


"""
struct NanoDataset
    length::Int64           # Number of tokens in the text 
    block_size::Int64       # How long an observation is. This is important for the dataloader
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

"""
    encode(d::NanoDataset, s::String)

Encode a string into tokens
"""
encode(d::NanoDataset, s::String) = [d.ch_to_int[ss] for ss in s]

"""
    decode(d::NanoDataset, i::AbstractArray{<:Integer}

Decodes a set of tokens into a string.
"""
decode(d::NanoDataset, i::AbstractArray{<:Integer}) = [d.int_to_ch[ii] for ii in i]


"""
    Base.length(d::NanoDataset)

Total number of tokens in the Dataset
"""
Base.length(d::NanoDataset) = d.length


function Base.getindex(d::NanoDataset, i::Int)
    1 <= i <= d.length - d.block_size || throw(ArgumentError("Index is out of bounds"))
    return d.data[i:i+d.block_size-1]
end



