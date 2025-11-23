using BenchmarkTools
using LinearAlgebra
using Lux
using NNlib
using Random
using Statistics

# We are operating on this torch tensor 
# 
"""
tensor([[[ 0.1808, -0.0700],
         [-0.3596, -0.9152],
         [ 0.6258,  0.0255],
         [ 0.9545,  0.0643],
         [ 0.3612,  1.1679],
         [-1.3499, -0.5102],
         [ 0.2360, -0.2398],
         [-0.9211,  1.5433]],

        [[ 1.3488, -0.1396],
         [ 0.2858,  0.9651],
         [-2.0371,  0.4931],
         [ 1.4870,  0.5910],
         [ 0.1260, -1.5627],
         [-1.1601, -0.3348],
         [ 0.4478, -0.8016],
         [ 1.5236,  2.5086]],

        [[-0.6631, -0.2513],
         [ 1.0101,  0.1215],
         [ 0.1584,  1.1340],
         [-1.1539, -0.2984],
         [-0.5075, -0.9239],
         [ 0.5467, -1.4948],
         [-1.2057,  0.5718],
         [-0.5974, -0.6937]],

        [[ 1.6455, -0.8030],
         [ 1.3514, -0.2759],
         [-1.5108,  2.1048],
         [ 2.7630, -1.7465],
         [ 1.4516, -1.5103],
         [ 0.8212, -0.2115],
         [ 0.7789,  1.5333],
         [ 1.6097, -0.4032]]])
"""

# This is a copy in Julia
x_py = cat(
    # First slice (:, :, 1)
    [
      0.1808 -0.3596  0.6258  0.9545  0.3612 -1.3499  0.2360 -0.9211;
      1.3488  0.2858 -2.0371  1.4870  0.1260 -1.1601  0.4478  1.5236;
     -0.6631  1.0101  0.1584 -1.1539 -0.5075  0.5467 -1.2057 -0.5974;
      1.6455  1.3514 -1.5108  2.7630  1.4516  0.8212  0.7789  1.6097
    ],
    # Second slice (:, :, 2)
    [
     -0.0700 -0.9152  0.0255  0.0643  1.1679 -0.5102 -0.2398  1.5433;
     -0.1396  0.9651  0.4931  0.5910 -1.5627 -0.3348 -0.8016  2.5086;
     -0.2513  0.1215  1.1340 -0.2984 -0.9239 -1.4948  0.5718 -0.6937;
     -0.8030 -0.2759  2.1048 -1.7465 -1.5103 -0.2115  1.5333 -0.4032
    ];
    dims=3
)

C, T, B = 2, 8, 4

# This is re-arranged to julia layout
x = permutedims(x_py, (3,2,1))

# The layout of this matrix is like
#
#
# x_11  x_12  x_13 ... x_1T
# x_21  x_22  x_23 ... x_2T
# ...   ..     ...      ...
# x_C1  x_C2  X_C3 ... X_CT
#
# Now we are looking at three ways to calculate the average of 
# vectors [x_11; x_21; ...; x_C1], [x_21; x_22; ...; x_C2], ... [x_i1; x_i2; ...; x_Ci]
# where i ≤ t. This is the running average along the time dimension.
# These averages are calculated individually for each batch.
# A function that calculates the x[c,t,b] = mean_{i<=t} x[c,i,b] at 
# https://youtu.be/kCc8FmEb1nY?t=2759
#
# 
#
#
# Trying to be nice with memory allocation
# Note: In torch, you can omit indices in a way you can't in Julia
# For example, in python x[0,0] means the same as x[0,0,:].
# You can index x with only 2 indices although it is 3 dimensional.
# Julia requires you to write out the third dimension
function token_avg_view(x)
    xbow = zeros(eltype(x), C, T, B) 
    for ix_b in 1:B                             # Pick out a single batch
        for ix_t in 1:T                         # For each token vector, iterating over time in the sequence
            xprev = view(x, :, 1:ix_t, ix_b)    # Create a view of all preceeding tokens including the current ont
            m = mean(xprev, dims=2)             # Calculate the mean over the sequence.
            xbow[:, ix_t, ix_b] .= m[:, 1]
        end
    end
    xbow
end






@benchmark token_avg_view(x)
"""
julia> @benchmark mean_thing_view(x)
BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
 Range (min … max):  23.291 μs …  9.777 ms  ┊ GC (min … max): 0.00% … 99.42%
 Time  (median):     27.125 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   28.536 μs ± 97.509 μs  ┊ GC (mean ± σ):  3.41% ±  0.99%

                    ▁▂▇█▆▃
  ▁▁▁▂▁▂▂▂▁▁▁▁▁▁▁▂▂▄██████▇▆▆▆▇▆██▇▆▄▅▄▃▃▂▂▂▂▂▁▂▂▁▂▂▂▂▂▂▁▁▂▁▁ ▃
  23.3 μs         Histogram: frequency by time        32.6 μs <

 Memory estimate: 32.89 KiB, allocs estimate: 716.
"""



function token_avg_noview(x)
    xbow = zeros(eltype(x), C, T, B) 
    for ix_b in 1:B
        for ix_t in 1:T
            # Here calling x[: 1:ix_t, ix_b] creates a copy of the vector.
            # We don't need that. Use views!
            m = mean(x[:, 1:ix_t, ix_b], dims=2)
            xbow[:, ix_t, ix_b] .= m[:, 1]
            #@show xprev
        end
    end
    xbow
end


@benchmark token_avg_noview(x)


"""
julia> @benchmark mean_thing_noview(x)
BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
 Range (min … max):  33.000 μs …  10.286 ms  ┊ GC (min … max): 0.00% … 99.32%
 Time  (median):     38.167 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   40.657 μs ± 134.711 μs  ┊ GC (mean ± σ):  4.65% ±  1.40%

                      ▃▄██▆▂▂
  ▁▁▁▁▁▂▁▂▁▁▁▂▁▁▁▁▂▂▃▆████████▇▆▅▅▅▆▅▆▆▅▄▅▄▄▃▃▃▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁ ▃
  33 μs           Histogram: frequency by time         44.9 μs <

 Memory estimate: 52.14 KiB, allocs estimate: 1324.
"""

xbow = token_avg_view(x)

# To create triangular matrces in Julia, we call the triu function, just as in torch.
a = triu(fill(true, 2,2))
#     / 1  1 \
# a = \ 0  1 /
#
#      2  7
# b =  6  4
#      6  5
#
#
#           2  (2 + 7)
# b * a =   6  (6 + 4)
#           6  (6 + 5)
b = [2.0 7.0; 6.0 4.0; 6.0 5.0]
c = b * a

# The second row of c should now be the column-wise sum of c[1,:], c[2,:]

# This is the row-wise sum, but we need to calculate the row-wise average.
# To do this, we can scale the rows of a by the row-wise sum.

# We can adapt this to calculate a mean from a sum
a = triu(fill(true, 2,2)) 
a = a ./ sum(a, dims=1) # Divide by the row-wise sum. We don't need keepdim=True, as in pytorch.
#
#
# a = / 1 0.5  \ 
#     \ 0  0.5 /
#
# Now the rows of c store the average of all the elements deposited in the row.
c = b * a




# The central idea is that we can formulate the mean_{i≤t} operation by using a lower triangular matrix.
# In particular
#
wts = triu(fill(true, T, T))
wts = wts ./ sum(wts, dims=1)

# Now we can express the time-average through matrix multiplication.
# We multiply the weights matrix for the first batch
xbow2 = x[:,:,1] * wts
xbow[:,:,1] ≈ xbow2

# Do do this for every batch in one go, we can use NNlib.batched_mul.
# Note that 
size(wts)
# (8, 8)
# and 
size(x)
# (2, 8, 4).
# We want to multiple all four (2,8) batches of x with the (8,8) wts matrix to get an 
# (2,8) * (8,8) = (2,8) matrix. To properly broadcast this in NNlib, we have to add another
# dimension to wts.
NNlib.batched_mul(x, reshape(wts, (8,8,1)))
# We can also use fancy notation, use ⊠ \boxtimes
# This is the final expression:
xbow2 = x ⊠ reshape(wts, (8,8,1)) 
xbow2 ≈ xbow

# To do weighted aggregations, it is useful to use a softmax.
# Like 1 and 0 are just extreme cases. In the future we want to allow intermediate values
# This softmax formulation here prepares us for this.
my_triu = triu(fill(true, T, T))
wts2 = zeros(T, T)
wts2[my_triu .!= 1.0] .= -Inf
wts2 = softmax(wts2, dims=1)
wts2 ≈ wts
xbow3 = x ⊠ reshape(wts2,(T,T,1))
xbow3 ≈ xbow

"""
julia> @benchmark mean_thing_mat(x)
BenchmarkTools.Trial: 10000 samples with 72 evaluations per sample.
 Range (min … max):  756.361 ns … 92.444 μs  ┊ GC (min … max): 0.00% … 98.72%
 Time  (median):     886.000 ns              ┊ GC (median):    0.00%
 Time  (mean ± σ):     1.017 μs ±  1.449 μs  ┊ GC (mean ± σ):  8.40% ±  8.99%

   █
  ▂█▅▅▃▂▂▂▂▂▂▂▁▁▁▁▁▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▂▂ ▂
  756 ns          Histogram: frequency by time         5.56 μs <

 Memory estimate: 3.61 KiB, allocs estimate: 22.
"""

function token_avg_mat(x)
    T = size(x, 2)
    my_triu = triu(ones(T, T))
    wts = zeros(T, T)
    wts[my_triu .!= 1.0] .= -Inf 
    wts = softmax(wts, dims=1)
    xbow = x ⊠ reshape(wts, (T, T, 1))
    xbow
end


# All three mean-thing functions are equal
xbow1 = token_avg_noview(x)
xbow2 = token_avg_view(x)
xbow3 = token_avg_mat(x)
xbow1 ≈ xbow2
xbow1 ≈ xbow2

#####


#############################################################################################
# Continue on implementing self-attention
# Video time-code:  https://youtu.be/kCc8FmEb1nY?t=3757
#

rng = Random.default_rng()
Random.seed!(rng, 1337)

C, T, B = 32, 8, 4

x = randn(Float32, C, T, B)

head_size = 16

# Every token emitts a key
key = Dense(C => head_size; use_bias=false)
ps_k, st_k = Lux.setup(rng, key)

# Every token emitts a query
query = Dense(C => head_size; use_bias=false)
ps_q, st_q = Lux.setup(rng, query)


value = Dense(C => head_size; use_bias=false)
ps_v, st_v = Lux.setup(rng, value)

k, _ = key(x , ps_k, st_k)  # size = (head_size, T, B)
q, _ = query(x, ps_q, st_q) # size = (head_size, T, B)
v, _ = value(x, ps_v, st_v) # size = (head_size, T, b)

# Instead of a uniform coupling matrix, we now use key-query dot-products.
# This makes the coupling data-driven.
wts_unrolled = [q[:,:,ix]' * k[:,:,ix] for ix in axes(q,3)]

# We write this with Boxtimes. (head_size, T, B) * ( T, head_size, B) -> (T, T)
#wts_box = q ⊠ permutedims(k, (2,1,3))
wts = permutedims(q, (2,1,3)) ⊠ k

wts_unrolled[1] == wts[:,:,1]
wts_unrolled[2] == wts[:,:,2]
wts_unrolled[3] == wts[:,:,3]
wts_unrolled[4] == wts[:,:,4]

wts2 = copy(wts)
wts3 = copy(wts)

my_triu = triu(ones(T, T))
mm = tril(fill(true, 16, 16), -1)

# Explore 3 different methods to remove non-causal coupling
# 1. Like in NanoGPT video, direct translation
function mask1(wts, tt)
    for ix_b ∈ axes(wts, 3)
        wts[tt.!= 1.0, ix_b] .= -Inf
    end
    wts
end

@benchmark mask1(wts, my_triu)
#BenchmarkTools.Trial: 10000 samples with 202 evaluations per sample.
# Range (min … max):  378.510 ns …  17.847 μs  ┊ GC (min … max): 0.00% … 96.14%
# Time  (median):     473.802 ns               ┊ GC (median):    0.00%
# Time  (mean ± σ):   528.838 ns ± 385.756 ns  ┊ GC (mean ± σ):  9.47% ± 12.71%
#
#   ▅██▇▄▂▁                                                      ▂
#  █████████▆▅▃▃▅▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃▄▃▃▄▁▄▃▅▄▆▆▅▆▆▆▇▆▆▇█▆▆▆▅▅▅▃▅▅ █
#  379 ns        Histogram: log(frequency) by time       2.19 μs <
#
#  Memory estimate: 2.50 KiB, allocs estimate: 20.

# 2. Double for loop, eliminates need for triu matrix
function mask2(wts)
    for ix_b ∈ axes(wts, 3)
        for j in 1:T
            for i in j+1:T
                wts[i, j, ix_b] = -Inf
            end
        end
    end
    return wts
end

@benchmark mask2(wts)

# 3. Don't index through triu, but subtraction should broadcast correctly
function mask3(wts, tt)
    causal_mask = view(tt, 1:T, 1:T)
    wts = wts .- (causal_mask .* 1e12)
    return wts
end

@benchmark mask3(wts, m) setup = (m = tril(fill(true, 16, 16), -1))
#BenchmarkTools.Trial: 10000 samples with 641 evaluations per sample.
# Range (min … max):  171.997 ns …  10.002 μs  ┊ GC (min … max):  0.00% … 96.20%
# Time  (median):     213.924 ns               ┊ GC (median):     0.00%
# Time  (mean ± σ):   279.923 ns ± 250.421 ns  ┊ GC (mean ± σ):  22.72% ± 20.89%
#
#   ▆█▇▄▁                                       ▁▁▂▂▂▁           ▂
#  ██████▇▇▅▇▇▆▄▁▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃▅▇██████▇█▇▇▆▇▆▆▆▆ █
#  172 ns        Histogram: log(frequency) by time        1.1 μs <
#
# Memory estimate: 2.36 KiB, allocs estimate: 8.
#


mask1(wts, my_triu)
mask2(wts)
mask3(wts, mm)

# All three ways of masking five the same output
mask1(wts, my_triu) == mask2(wts)
mask1(wts, my_triu) == mask3(wts, mm)




# Implementation of sha with mask1
function single_head_attention(x, head_size)
    C, T, B = size(x)
    # Models for key, query, and value
    key = Dense(C => head_size; use_bias=false)
    ps_k, st_k = Lux.setup(rng, key)

    query = Dense(C => head_size; use_bias=false)
    ps_q, st_q = Lux.setup(rng, query)

    value = Dense(C => head_size; use_bias=false)
    ps_v, st_v = Lux.setup(rng, query)

    k, _ = key(x , ps_k, st_k)  # size = (head_size, T, B)
    q, _ = query(x, ps_q, st_q) # size = (head_size, T, B)
    v, _ = value(x, ps_v, st_v) # size = (head_size, T, B)

    # Bag-of-word averaging is replaced by q*k vector products
    wts = permutedims(q, (2,1,3)) ⊠ k # size = (T, T)

    # Auto-regressive structure implemented by triu matrix, prohibiting communication
    # with the past
    my_triu = triu(ones(T, T))
    # Unfortunately, we don't have a batched masked fill. So iterate it out
    for ix_b ∈ axes(wts, 3)
        wts[my_triu .!= 1.0, ix_b] .= -Inf 
    end
    # Normalize so we get a probability
    wts = softmax(wts, dims=1)
    # Instead of working with token embeddings directly, work with the value
    v ⊠ wts
end



# Implementation of sha with mask3
function sha_v2(x, head_size)
    C, T, B =size(x)
    key = Dense(C => head_size; use_bias=false)
    ps_k, st_k = Lux.setup(rng, key)

    query = Dense(C => head_size; use_bias=false)
    ps_q, st_q = Lux.setup(rng, query)

    value = Dense(C => head_size; use_bias=false)
    ps_v, st_v = Lux.setup(rng, query)

    k, _ = key(x , ps_k, st_k)  # size = (head_size, T, B)
    @show size(k)
    q, _ = query(x, ps_q, st_q) # size = (head_size, T, B)
    @show size(q)
    v, _ = value(x, ps_v, st_v) # size = (head_size, T, B)
    @show size(v)

    # Bag-of-word averaging is replaced by q*k vector products
    wts = permutedims(q, (2,1,3)) ⊠ k # size = (T, T)

    causal_mask = tril(fill(true, T, T), -1)

    wts = wts .- (causal_mask .* 1f12)
    # Normalize so we get a probability
    wts = softmax(wts, dims=1)
    @show size(wts)
    # Instead of working with token embeddings directly, work with the value
    out = v ⊠ wts 
    @show size(out)
    out


end


# Test that both sha implementations give the same answer
Random.seed!(32)
out_v1 = single_head_attention(Float32.(x), 16)

Random.seed!(32)
out_v2 = sha_v2(Float32.(x), 16)

out_v1 ≈ out_v2



# Test how we can use batched matrix multiplication to speed up calculation of attention scores

function calc_scaled_dot_prod_attn(Q, K, V)
    xT = eltype(Q)
    head_size = size(Q, 1)
    T = size(Q, 2)
    wts = permutedims(K, (2, 1, 3)) ⊠ Q ./ xT(sqrt(head_size))
    # For large values, softmax may converge to a one-hot vector.
    # If we scale them, the resulting distribution after softmax will be more diffuse
    causal_mask = tril(fill(true, T, T), -1)
    wts = wts .- (causal_mask .* 1f12) # size (T, T, B)
    wts = softmax(wts, dims=1)
    V ⊠ wts
end


function mha_v1(x, head_size, n_heads)
    n_embd  = size(x, 1)
    key = Dense(n_embd => n_heads * head_size; use_bias=false)
    ps_k, st_k = Lux.setup(rng, key)

    query = Dense(n_embd => n_heads * head_size; use_bias=false)
    ps_q, st_q = Lux.setup(rng, query)

    value = Dense(n_embd => n_heads * head_size; use_bias=false)
    ps_v, st_v = Lux.setup(rng, query)

    k, _ = key(x , ps_k, st_k)  # size = (head_size, T, B)
    q, _ = query(x, ps_q, st_q) # size = (head_size, T, B)
    v, _ = value(x, ps_v, st_v) # size = (head_size, T, B)
    #@show size(k), size(q), size(v)

    # Reshape to split into different heads
    q_rs = reshape(q, (head_size, n_heads, size(q)[2:end]...));
    k_rs = reshape(k, (head_size, n_heads, size(q)[2:end]...));
    v_rs = reshape(v, (head_size, n_heads, size(q)[2:end]...));
    #@show size(q_rs), size(k_rs), size(v_rs)

    attn_scores = [calc_scaled_dot_prod_attn(q_rs[:, i, :, :], k_rs[:, i, :, :], v_rs[:, i, :, :]) for i in axes(q_rs, 2)]

    return cat(attn_scores..., dims=1)
end

function mha_v2(x, head_size, n_heads)
    n_embd, seq_len, num_batch  = size(x)
    key = Dense(n_embd => n_heads * head_size; use_bias=false)
    ps_k, st_k = Lux.setup(rng, key)

    query = Dense(n_embd => n_heads * head_size; use_bias=false)
    ps_q, st_q = Lux.setup(rng, query)

    value = Dense(n_embd => n_heads * head_size; use_bias=false)
    ps_v, st_v = Lux.setup(rng, query)

    k, _ = key(x , ps_k, st_k)  # size = (head_size, T, B)
    q, _ = query(x, ps_q, st_q) # size = (head_size, T, B)
    v, _ = value(x, ps_v, st_v) # size = (head_size, T, B)

    # Reshape to split into different heads
    q_rs = reshape(q, (head_size, n_heads, size(q)[2:end]...));
    k_rs = reshape(k, (head_size, n_heads, size(q)[2:end]...));
    v_rs = reshape(v, (head_size, n_heads, size(q)[2:end]...));


    # Now the trick: combine n_heads with the batch dimension (if it exists).
    if ndims(q_rs) == 4 # case (h, n, T, B) h: head_size. n: num_heads. T: sequence length. B: Batch
        # permute dimensions: (h, n, T, B) -> (h, T, n, B)
        q_p = permutedims(q_rs, (1, 3, 2, 4))
        k_p = permutedims(k_rs, (1, 3, 2, 4))
        v_p = permutedims(v_rs, (1, 3, 2, 4))
        # now merge the head and batch dimension together.
        q_b = reshape(q_p, (head_size, seq_len, :))
        k_b = reshape(k_p, (head_size, seq_len, :))
        v_b = reshape(v_p, (head_size, seq_len, :))
    else
        # If there is no batch dimension, use the head dimension as the batch dimension
        # (h, n, T) -> (h, T, n)
        q_b = permutedims(q, (1, 3, 2))
        k_b = permutedims(k, (1, 3, 2))
        v_b = permutedims(v, (1, 3, 2))
    end

    # Calculate attention
    attn_b = calc_scaled_dot_prod_attn(q_b, k_b, v_b)

    # Reshape and permute to original layout
    if ndims(q_rs) == 4
        # (h, T, n*B) -> (h, T, n, B)
        attn_q_p = reshape(attn_b, (head_size, seq_len, n_heads, B))
        # (h, T, n, B) -> (h, n, T, B)
        attn_rs = permutedims(attn_q_p, (1, 3, 2, 4))
        # (h, n, T, B) -> (h*n, T, B)
        attn = reshape(attn_rs, (n_embd, seq_len, num_batch))
    else 
        # (h, T, n) -> (h, n, T)
        attn_rs = permutedims(attn_b, (1, 3, 2))
        # (h, n, T) -> (h*n, T)
        attn = reshape(attn_rs, (n_embd, seq_len))
    end

    return attn
end


n_embd = 64
n_heads = 4
head_size = n_embd ÷ n_heads
T = 64
B = 32

Random.seed!(1337)
x = randn(Float32, n_embd, T, B);

Random.seed!(1337)
x_1 = mha_v1(x, 16, 4)

Random.seed!(1337)
x_2 = mha_v2(x, 16, 4)
   
x_1 ≈ x_2

@benchmark mha_v1(x, 16, 4) setup=(x=randn(Float32, n_embd, T, B))
@benchmark mha_v2(x, 16, 4) setup=(x=randn(Float32, n_embd, T, B))

