
using ADTypes
#using Enzyme
using Distributions
using Lux
using OneHotArrays
using Optimisers
using Random
using Zygote




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

    data = encode(join(lines, ""))
    return data, encode, decode, vocab_size
end

dataset, encode_fun, decode_fun, vocab_size = load_dataset("data/input.txt")



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

function loss_fun(model, ps, st, (xb, yb))
    # Outputs are interpreted as the logits.
    logits, _ = model(xb, ps, st)
    (C, T, B) = size(logits)
    logits_t = reshape(logits, C, T*B)  # Reshape, so that dim2 is along separate samples
    yb_t = reshape(yb, T*B)
    # Here we have to resort to OneHotArrays to get the CrossEntropy to work.
    oh = onehotbatch(yb_t, 1:65)
    # USe a negative log-likelihood loss function
    loss = CrossEntropyLoss(; agg=mean, logits=Val(true))(logits_t, oh)
    return loss, st, NamedTuple()
end



N = length(dataset)
N_train = Int(round(0.9 * N))
data_train = dataset[1:N_train]
data_test = dataset[N_train+1:end]

 
rng = Random.default_rng()
Random.seed!(rng, 1337)

model = Embedding(vocab_size => vocab_size)
ps, st = Lux.setup(rng, model)
opt = Adam(0.03f0)

# TrainState is a useful struct defined by lux. It is essentially a warpper over parameters, state, optimizer state,
# and the model. We only need to pass this into our training function
tstate = Training.TrainState(model, ps, st, opt)
vjp_rule = AutoEnzyme()

function train(tstate::Training.TrainState, vjp, data_set, num_epochs)
    for epoch in 1:num_epochs
        xb, yb = get_batch(rng, data_set, 4, 8)
        _, loss, _, _tstate = Training.single_train_step!(vjp, loss_fun, (xb, yb), tstate)
        println("Epoch: $(epoch)    Loss: $(loss)")
    end
    return tstate
end


train(tstate, AutoZygote(), data_train, 10)

