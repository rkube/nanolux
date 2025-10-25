using Lux
using Random

# Following along
#
#
# First, read the dataset 
lines = readlines("data/input.txt")
# This 
"""julia-repl
julia> lines = readlines("data/input.txt")
40000-element Vector{String}:
"""
# This is a vector of 40_000 strings, where each string represents a line.
# To get the number of characters, calculate the length of each sum
sum([length(s) for s in lines])
# The first 20 lines are
lines[1:20]

# One thing that will come in handy later is to remove empty lines in the this vector

# To build our vocabulary, we want the find the unique characters occuring in
# all strings. With our dataset represented as a vector of strings,
# we need to find all characters in each line and repeat that for all lines.
# Julia allows to concisively represent this double for loop as
_out = [c for l ∈ lines for c ∈ l]

# Note that this notation puts the inner for loop, for c ∈ l where iteration is over
# a single line, last.
# Now we can identify all unique characters with the `unique` function:
chars = sort!(unique(_out))
# A small subtlety is that since we built our vocabulary from lines, it is missing the newline
# character. Let's push it in:
push!(chars, '\n')

# Note that in Julia chars are identified by single ticks (') and strings are identified by (").


# This is a Vector{Char}, a vector of char. We find it's length by calling length
vocab_size = length(chars)


# Julia's repl prints adds type-information about the charset by default when
# printing characters:
# '$': ASCII/Unicode U+0024 (category Sc: Symbol, currency)
# Here it shows that it is a ASCII/Unicode character and gives information about the group
# that character is in.
# Printing the vocabulary as a string
String(chars)
# displays all characters in sequence, as expected for a String.
# We have a white space first, followed by special characters and digitis, and finally 
# uppercase and lowercase letters of the alphabet at the end.

# ## Tokenizing
# Next, we develop a strategy to tokenize the text. This is the process of converting
# the characters in a text to a sequence of integers by well-defined rules.
#
# Julia provides the enumerate function that counts the numbers of iteration in a for
# loop. For example
"""julia-repl
julia> for (ix, val) in enumerate(4:8)
       @show ix, val
       end
(ix, val) = (1, 4)
(ix, val) = (2, 5)
(ix, val) = (3, 6)
(ix, val) = (4, 7)
(ix, val) = (5, 8)
"""

# We can use this syntax inside the constructor of a dictionary like this
int_to_ch = Dict( ix => val for (ix, val) in enumerate(chars) )
ch_to_int = Dict( val => ix for (ix, val) in enumerate(chars) )

# As you see, the syntax for defining Dictionaries is Dict(key => value) and we 
# can use inline for loop inside the constructor.
# Julia being a typed language also correctly picks up the types in our dictionary.
# The Dictionary int_to_ch is of type Dict{Int64, Char}, denoting that keys are of type 
# Int64 and values are of type Char. For the dictionary ch_to_int the order is reversed.
"""julia-repl
julia> int_to_ch = Dict( ix => val for (ix, val) in enumerate(chars) )
Dict{Int64, Char} with 64 entries:
[...]
"""

# Defining functions that encode a string, mapping the individual characters to the
# code can be defined like so
encode(str) = [ch_to_int[s] for s in str]
decode(tok) = [int_to_ch[t] for t in tok]

# When we apply the encode function to a string, the function returns a vector of the
# tokens - the encoded values of each character.
encode("Hi there")
# Calling decode on this vector returns the original input, as a Vector{Char}.
decode(encode("Hi there"))

# Unlike python, Julia doesn't require special data types to be used in deep learning.
# So to encoding the entire dataset into a vector is fine. We can do this by calling
data = encode(join(lines, ""))

# Here we call the join function, which concatenates all the individual strings of the
# lines vector into one long string. The output is a 10775394-element Vector of type Int64
"""julia-repl
julia> encode(join(lines, ""))
1075394-element Vector{Int64}:
 18
 47
 56
[...]
"""

# To split the data into training and validation set, we split the vector using indexing
# notation
#
N = length(data)
N_train = Int(round(0.9 * N))
data_train = data[1:N_train]
data_test = data[N_train+1:end]

# One notable difference in the slice notation is that julia requires to give the full range.
# Writing just data[:10] will give you the 10-th element of the vector. And omitting `end`
# throws a ParseError.
# Another difference is that Julia's indexing is 1-based. This is natural for counting
# elements. If you ever opened a Linear Algebra book you remember that matrix elements are
# counted started from 1. In this spirit, Julia's syntax is close to written math.
#
# To split the data into block sizes, we define a block size and look at the first 8 trainnig
# data
block_size = 8
data_train[1:block_size+1]

# Here using the syntax data_train[1:block_size+1] to get the first 9 characters.
# The task of the transformers, data_train[1:block_size] will be used to predict data_train[block_size+1]
# As in the video, we are spelling it out in code. In my personal opinion, the array indexing
# syntax feels more natural to write. We are using the first t tokens to predict token t+1.

for t ∈ 1:block_size
    context = data_train[1:t]
    target = data_train[t+1]
    println("When input is $(context) the target is $(target)")
end

# TODO: Talk about rng initialization
# For the next part, we are going to use random numbers. Julia provides random number generators through the 
# [Random](https://docs.julialang.org/en/v1/stdlib/Random/) package. To re-create reproducible random numbers
# we specify the RNG we use and use a fixed seed. This way, everytime we restart this script and draw random
# numbers, the sequence will be identical.
rng = Random.default_rng()
Random.seed!(rng, 1337)

# Loading data
# As discussed in the previous section, data will be loaded in blocks. Their length give the context length of
# the model. But we also want to use data parallelism. This is, when the model processes independent batches of
# data. These will not interact with one another. But processing them in parallel saves time.
#
# The julia implementation looks like this:
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

# There are two note-worthy differences to the python implementation. First, we are passing the dataset
# to the function, besides the batch_size and the block_size. This is motivated by the way Julia compiles
# functions. 
# the indices for the 
"""julia-repl
julia> x, y = get_batch(data_train, 4, 8)
([43 54 42 41; 53 53 53 53; … ; 1 44 53 56; 54 1 1 6], [53 53 53 53; 54 57 52 52; … ; 54 1 1 6; 43 46 63 1])

julia> x
8×4 Matrix{Int64}:
 43  54  42  41
 53  53  53  53
 54  57  52  52
 50  43   6  55
 43   1   1  59
 11  53  58  43
  1  44  53  56
 54   1   1   6
"""

# Let's see how the example above works when we make it more complicated by including the batch size.
#
batch_size = 4
xb, yb = get_batch(data_train, batch_size, block_size)


# The loop below prints out the conext and the target, as provided by the data loaders.
# Note that we adapted a more julian way to iterate over Arrays. Instead of writing 
# `for b in 1:batch_size` we use the `axes` function. This function provides an iterator
# over the selected dimension of the Array.
# Julia stores array in column-major order. This means that the elements xb[i, b] and xb[i+1,b]
# are consecutive in memory. By ordernig the nested for loops so that the innermost loop
# iterates over the first dimension and the outermost loop iterates over the second dimension of
# xb, we make sure that the over the course of both iterations we are traversing the memory space
# of the array in order.
for b in axes(xb, 2)
    for t in axes(xb, 1)
        context = xb[1:t, b]
        target = yb[t, b]
        println("Context = $(context) target = $(target)")
    end
end


# Now, we create a Lux model with an embedding layer. This is our first contact with Lux.jl
# Say something about how parameters and state are not within the structs. So we typically have
# to pass them in every call
m = Embedding(vocab_size => vocab_size)
ps, st = Lux.setup(rng, m)

# This output is of shape (65, 8, 4). Again, since Julia uses column-major data storage,
# the order of the dimensions is reversed compared to python.
#
# To apply the model to a batch of training data, simply call it and remember to pass the 
# parameters and the state. The output is of shape (num_classes, block_size, batch_size) ≡ (C, B, N)
out, _ = m(xb, ps, st)

# Julia defines a function-like interface. So instead of evaluating the loss function inside a forward pass,
# we simply evaluate it outside:
#
function loss_fun(model, ps, st, (xb, yb))
    # Outputs are interpreted as the logits.
    logits, _ = model(xb, ps, st)
    (C, T, B) = size(logits)
    logits_t = reshape(logits, C, T*B)  # Reshape, so that dim2 is along separate samples
    # USe a negative log-likelihood loss function
    loss = CrossEntropyLoss(; agg=mean, logits=Val(true))(logits, yb)
    return loss
end

loss_fun(m, ps, st, (xb, yb))


