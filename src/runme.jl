using ADTypes
using Zygote
using Distributions
using Lux
using OneHotArrays
using Optimisers
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
# data. The block size is called T, and defines the size of the context window.
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
batch_size = 4 # This is called B in short.
xb, yb = get_batch(rng, data_train, batch_size, block_size)


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
model = Embedding(vocab_size => vocab_size)
ps, st = Lux.setup(rng, model)

# The model embeds each token in a (C=vocab_size)-dimensional layer. 
# As a result, the model transforms its input xb of size(T, B) = (block_size, batch_size) into
# a (C=65, T=8, B=4)-size matrix. Since Julia uses column-major layout, the embedding vectors are
# stored consecutively in memory.
#
# To apply the model to a batch of training data, simply call it and remember to pass the 
# parameters and the state. The output is of shape (num_classes, block_size, batch_size) ≡ (C, B, T)
out, _ = model(xb, ps, st)

# Julia defines a function-like interface. So instead of evaluating the loss function inside a forward pass,
# we simply evaluate it outside.
# In addition, this loss function adheres to the [Training API](https://lux.csail.mit.edu/stable/api/Lux/utilities)
# One particularity of this API is that loss functions can take inputs in the form of 
# loss_fun(model, ps, st, (x,y)) and return the loss, updated states, and an empty named tuple.
#
# We also have to do some reshaping. 
#
#
function loss_fun(model, ps, st, (xb, yb))
    # Outputs are interpreted as the logits.
    logits, _ = model(xb, ps, st)
    (C, T, B) = size(logits)
    logits_t = reshape(logits, C, T*B)  # Reshape, so that dim2 is along separate samples
    yb_t = reshape(yb, T*B)
    # Here we have to resort to OneHotArrays to get the CrossEntropy to work.
    oh = onehotbatch(yb_t, 1:C)
    # USe a negative log-likelihood loss function
    loss = CrossEntropyLoss(; agg=mean, logits=Val(true))(logits_t, oh)
    println("loss_fun: loss = $(loss)")
    return loss, st, NamedTuple()
end

# Now we can evaluate the loss function. The untrained model should be random. We have 65 possible vocabulary
# elements. For random choices, we would expect something in the order of log(1.0/65.0)
loss_fun(model, ps, st, (xb, yb))

# Now we want to have a function to generate samples from the data.
#
#


"""
Helper function for batched categorical sampling.
`probs` is a (Classes, Batch) matrix.
Returns a (1, Batch) matrix of sampled indices.
"""
function sample_categorical_batch(probs::AbstractMatrix)
    C, B = size(probs)
    # Pre-allocate output
    idx_next = Matrix{Int}(undef, 1, B)
    
    # Normalize each column to be safe for Distributions.Categorical
    # This prevents errors if sum(col) is 0.999999
    probs_normalized = probs ./ sum(probs; dims=1)
    
    for i in 1:B
        # view(probs_normalized, :, i) is a 1D view of the i-th column
        dist = Distributions.Categorical(probs_normalized[:, i])
        idx_next[1, i] = rand(dist)
    end
    
    return idx_next
end


"""
    generate(input, max_new_tokens, model, ps, st)

Autoregressively generates `max_new_tokens` new tokens based on the provided `input` context.

At each step, the entire sequence of tokens generated so far is passed to the `model` to predict 
the next token. This process is repeated `max_new_tokens` times.

# Arguments
- `input`: A `(T, B)` matrix of token indices representing the initial context, where `T` is the sequence length and `B` is the batch size.
- `max_new_tokens`: The number of new tokens to generate for each sequence in the batch.
- `model`: The language model to use for generation.
- `ps`: The parameters of the model.
- `st`: The state of the model.

# Returns
- A `(T + max_new_tokens, B)` matrix containing the original `input` context followed by the newly generated tokens.
"""
function generate(input, max_new_tokens, model, ps, st)
    # Pre-allocate an array with the current number of tokens + new number of tokens
    T, B = size(input)
    # Remember, T is the context. B are the batches that are run individually
    # 1. Pre-allocate the output array - this is of filesize
    tokens_out = similar(input, T + max_new_tokens, B)
    # 2. Get the initial context (the very last input token).
    tokens_out[1:T, :] .= input

    # 3. Get the initial context (the very last input token)
    current_token = view(tokens_out, 1:T, :)
    for ix_t in 1:max_new_tokens
        # 1. Get the logits from the model.
        #    Here we pass only the last token,
        logits, st = model(current_token, ps, st)
        #@show size(logits)

        # Apply softmax to the embedding dimension get probabilities, but only to the new prediction.
        # In particular, summing over the embedding dimension should give 1!
        # sum(probs, dims=1) = ones(1, B)
        # Apply softmax to get probabilities
        # 2. Get the logits for the last time step
        logits_last = logits[:, end, :] # Size (C, B)

        # 3. Convert the logits to a probability by applying softmax
        probs = softmax(logits_last; dims=1) # Shape (C, B)

        # 4. Sample one index for each batch item
        #    TODO: The current implementation won't work on the gpu. 
        #    To fix this, we'd need the gumbel-max trick.
        idx_next = sample_categorical_batch(probs) # (1, 
        #@show size(idx_next)

        # 5. Store the new token in the output array 
        tokens_out[T+ix_t:T+ix_t, :] .= idx_next 

        # 6. Update the context window for the next iteration 
        current_token = view(tokens_out, 1:T+ix_t, :)
    end
    return tokens_out
end

#Note for future me: Look into sampling the categorical distribution using GumbelSoftmax: https://casmls.github.io/general/2017/02/01/GumbelSoftmax.html

out = generate(xb, 3, model, ps, st)

# Generate output of the model for a simple input
out = generate(ones(Int64, 1, 1), 100, model, ps, st)

# Now we can also print the output generated by the model
print(join(decode(out), ""))


### Next, we'd have to train this model.
#
model = Embedding(vocab_size => vocab_size)
ps, st = Lux.setup(rng, model)
opt = Adam(0.03f0)

# TrainState is a useful struct defined by lux. It is essentially a warpper over parameters, state, optimizer state,
# and the model. We only need to pass this into our training function
tstate = Training.TrainState(model, ps, st, opt)

function train(tstate::Training.TrainState, vjp, data_set, num_epochs)
    for epoch in 1:num_epochs
        xb, yb = get_batch(rng, data_set, 4, 8)
        _, loss, _, _tstate = Training.single_train_step!(vjp, loss_fun, (xb, yb), tstate)
        println("Epoch: $(epoch)    Loss: $(loss)")
    end
    return tstate
end


train(tstate, AutoZygote(), data_train, 10_000)


out = generate(ones(Int64, 1, 1), 500, model, ps, st);
print(join(decode(out), ""))


# The mathematical trick in self-attention
#

Random.seed!(rng, 1337)

B, T, C = 4, 8, 2
x = randn(C, T, B)
size(x)

# We have 8 tokens in a batch that are not talking to each other.
# Now we make them talk to each other.
# Couple them in an autoregressive way. Token in location 5 should only communicate 
# with tokens 1..4. But not with 6..8. So we don't get any information from the future.
#
# Simplest way is to give token 5 an average over tokens 1..4. This is very lossy, but gets the
# point across.
#
# For every batch element calculate the average of the preceeding tokens.
# See notes on implementation in examples/tensor_thingies.jl
xbow = zeros(C, T, B) # Bag-of-words is when you just average up things.
for ix_t in 1:T
    for ix_b in 1:B
        # The corresponding pytorch expression is x[b, :t+1]. 
        # Pytorch implicitly expands this to x[b, :t+1, :]. In Julia there is no such implicit 
        # slicing. We have to select the channel dimension explicitly.
        xprev = view(x, :, 1:ix_t, ix_b)   # Select slice (C, t, b) as a view 
        m = mean(xprev, dims=2)
        xbow[:, ix_t, ix_b] .= m[:, 1]
    end
end

