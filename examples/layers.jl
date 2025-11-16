#
# Writing models in Lux can be done in multiple ways
#
#
#
using Lux
using MLUtils
using Random
using Reactant
using NanoLux


function get_model(vocab_size, n_embd, head_size)
    num_heads = n_embd ÷ head_size
    model = @compact(token_embedding = Embedding(vocab_size => n_embd),
                     pos_embedding = Embedding(vocab_size => n_embd),
                     sa_head = SingleHeadAttention(n_embd, n_embd),
                     ma_head = Parallel(vcat, [SingleHeadAttention(n_embd, head_size) for _ in 1:num_heads]...),
                     ffwd = Dense(n_embd => n_embd, relu),
                     lm_head = Dense(n_embd => vocab_size, relu)
                     ) do x
                     T, B = size(x)     # T: block_size (the sequence length), B: batch_size
                     tok_emb = token_embedding(x)   # size (C, T, B)
                     pos_emb = pos_embedding(1:T)   # size (C, T)
                     x = tok_emb .+ pos_emb
                     x = ma_head(x)
                     x = ffwd(x)
                     logits = lm_head(x)
                     @return logits
                     end
    return model
end

rng = Random.default_rng()
Random.seed!(rng, 1337)
batch_size = 3
block_size = 32
head_size = 16
n_embd = 64

vocab_size = 65 

x = rand(1:65, block_size, batch_size);
size(x)

# Let's define a simple embedding model. The input is (block_size, batch_size).
# This layers takes each token, an integer number, and embedds it into a n_embd-dimensional
# vector.
model_e = Embedding(vocab_size => n_embd)
ps_e, st_e = Lux.setup(rng, model_e)
x_e, _ =model_e(x, ps_e, st_e)
# The output is now of size (n_embd, block_size, batch_size)
size(x_e)

# The single attention head we implemented maps each embedding vector (there are (block_sizexbatch_size) of them)
# into n_embd dimensional vectors through Q, K, and V.
#
model_sha = SingleHeadAttention(n_embd, n_embd)
ps_s, st_s = Lux.setup(rng, model_sha)
x_sha, _ = model_sha(x_e, ps_s, st_s)
size(x_sha)

# In multi-head attention, the output for Q, K, and V is of dimension n_head. That is, Q, K, and V now map into
# n_head dimensional space. In this example, n_embd=64 and setting num_heads=4, the output of each head is 
# 16-dimensional. We can implement this easily by running the single-head attention layers in parallel. This is
# done by using a Parallel layer, which passes the input to each layer in the list argument.
# While we can specify a reduction, for example concatenation, let's just look at the output shape for each
# layer in the list:
#
num_heads = 4
head_size = n_embd ÷ num_heads
model_mha = Parallel(nothing, [SingleHeadAttention(n_embd, head_size) for _ in 1:num_heads]...)

ps_mha, st_mha = Lux.setup(rng, model_mha)
x_out, _ = model_mha(x_e, ps_mha, st_mha)

# When we now run the model, where we have to take the token emebddings `x_e` as input, we get a vector of length
# `num_heads` in return. Each item in this vector is the output of a single head attention layer with size 
# (head_size, block_size, batch_size).
size(x_out[1])
[size(x_out[i]) for i in 1:num_heads]




# A full transformer block consists of multiple attention heads, whose output goes into 
# a feed-forward block. For reusability, we may want to build separate Layer structures for this.
# We now explore how Lux can be used to define these custom layers.
# The easiest way to do this is by using the @compact macro 

model_mha = @compact(
        heads = Parallel(vcat, [SingleHeadAttention(n_embd, head_size) for _ in 1:num_heads]...),
        proj = Dense(n_embd => n_embd),
        dropout = Dropout(0.0)
    ) do x
    x = heads(x)
    x = dropout(proj(x))
    @return x
end

ps_mha, st_mha = Lux.setup(rng, model_mha)
x_mha, _ = model_mha(x_e, ps_mha, st_mha)
size(x_mha)

# To build a transformer block, we will also run the output of the multi-head attention through a feed-forward block.
# This is just a combination of multiple linear layers with additional dropout. Again, we can use the @compact
# macro to quickly specify this block:

model_ffwd = @compact(
        linear_1 = Dense(n_embd => 4 * n_embd, relu),
        linear_2 = Dense(4 * n_embd => n_embd),
        dropout = Dropout(0.0)
    ) do x
    x = linear_1(x)
    x = linear_2(x)
    x = dropout(x)
    @return x
end

# Instantiate the ffwd model and run it:
ps_ffwd, st_ffwd = Lux.setup(rng, model_ffwd)

x_ffwd, _ = model_ffwd(x_mha, ps_ffwd, st_ffwd)

size(x_ffwd)

