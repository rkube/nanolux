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
using NNlib


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
model_sha = SingleHeadAttention(n_embd, head_size)
ps_sha, st_sha = Lux.setup(rng, model_sha)
x_sha, _ = model_sha(x_e, ps_sha, st_sha)
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


# Now the kicker: We can be smart by doing the Q,K,V projections in a single matrix multiply.
#
# In single-head attention, Q, K, and V are projections from (n_embd) => (head_size).
# That is, they just calculate matrix multiplications like this:
x_out, _ = model_sha.query(x_e, ps_sha.query, st_sha.query)
x_out ≈ ps_sha.query.weight ⊠ x_e

# Here we use ⊠ to batch the matrix multiplication over the last dimension. For the first batch
# this is would be
x_out[:,:,1] ≈ ps_sha.query.weight * x_e[:,:,1]

# The size of the W_Q is just (head_size, n_embd). If we vertically stack W_Q with another W_Q, we get a
# matrix of size (2 * head_size, n_embd). In this way, we can calculate Q for multiple heads using only
# a single call to the matrix multiplication.
w = randn(head_size, n_embd)
q = w * x_e[:,:,1]

# We now stack w on top of each other. This effectively means we are using two heads:
w2 = vcat(w, w)
q2 = w2*x_e[:,:,1]

# The operation above is the identical to calculating q twice, one time for each head.
q2 ≈ vcat(q,q)

# We can no reshape q2 into the query for both heads. To do this, we introduce a new dimension after the first,
# which indices the individual attention heads.
q2_rs = reshape(q2, (head_size, 2, block_size));
q2_rs[:, 1, :] == q2_rs[:, 2, :]

# So what we are going to do for the multi-head implementation is to calculate Q, K, and V for all
# heads in one go. Then we are using reshaping to separate the concatenated Q,K,V into individual heads

model_mha = NanoLux.MultiHeadAttention(n_embd, 4)

ps_mha, st_mha = Lux.setup(rng, model_mha)

x_mha, _ = model_mha(x_e, ps_mha, st_mha)


# Test the feed-forward block
model_ffwd = NanoLux.FeedForward(64, 0.25)
ps_f, st_f = Lux.setup(rng, model_ffwd)

x_f, st_f = model_ffwd(x_mha, ps_f, st_f)

model_ln = LayerNorm((64,32))
ps_l, st_l = Lux.setup(rng, model_ln)


x_n, st_n = model_ln(x_f, ps_l, st_l)
