
export SingleHeadAttention
export MyMultiHeadAttention
export FeyedForwar
export Transformer

"""
    SingleHeadAttention

This stores the three dense layers needed to calculate single head attention
"""
struct SingleHeadAttention{Q, K, V} <: LuxCore.AbstractLuxContainerLayer{(:query, :key, :value)}
    query::Q 
    key::K 
    value::V
end

function SingleHeadAttention(in_dim::Int, head_size::Int; init_weight=glorot_uniform) 
    return SingleHeadAttention(
        Dense(in_dim => head_size; use_bias=false, init_weight=init_weight),
        Dense(in_dim => head_size; use_bias=false, init_weight=init_weight),
        Dense(in_dim => head_size; use_bias=false, init_weight=init_weight),
    )
end



function LuxCore.initialstates(::AbstractRNG, ::SingleHeadAttention{Q, K, V}) where {Q, K, V} 
    return (query=NamedTuple(;), key = NamedTuple(;), value=NamedTuple(;))
end


function (sha::SingleHeadAttention)(x::AbstractArray, ps, st::NamedTuple)
    # x has shape (features, sequence_length, batch_size)
    q, st_q = sha.query(x, ps.query, st.query)  # size = (head_size, T, B)
    k, st_k = sha.key(x, ps.key, st.key)        # size = (head_size, T, B)
    v, st_v = sha.value(x, ps.value, st.value)  # size = (head_size, T, B) 
    scores = calc_scaled_dot_prod_attn(q, k, v)
    scores, (query = st_q, key = st_k, value = st_v)
end

"""
    calc_scaled_dot_prod_attn(Q, K, V)

Calculate the scaled dot-product attention:

softmax(Q K^T / sqrt(d_k)) * V

# Arguments
- `Q` - Query tensor, of dimension (head_size, T, B)
- `K` - Key tensor, of dimension (head_size, T, B)
- `V` - Value tensor, of dimension (head_size, T, B)

# Returns 
- Attention scores 
   
"""
function calc_scaled_dot_prod_attn(Q, K, V)
    xT = eltype(Q)
    head_size = size(Q, 1)
    T = size(Q, 2)

    wts = permutedims(K, (2, 1, 3)) ⊠ Q ./ xT(sqrt(head_size))
    #wts = permutedims(k, (2, 1, 3)) ⊠ q ./ xT(sqrt(head_size)) # size = (T,T)

    #         <k1, q1>  <k1, q2>  ... <k1, qT>
    #         <k2, q1>  <k2, q2>  ... <k2, qT>
    #            ...      ...     ...   ...
    #  wts =  <kT, q1>  <kT, q2>  ...  <kT, qT>
    #
    # The mask needs to prevent a token at a given position to attending to subsequent tokens.
    # That is, <k_i, q_j> for i > j. These scalar products are in the lower triangular, below
    # the main diagonal. A lower triangular matrix from diagonal -1, does the trick.

    # For large values, softmax may converge to a one-hot vector.
    # If we scale them, the resulting distribution after softmax will be more diffuse
    causal_mask = tril(fill(true, T, T), -1)
    wts = wts .- (causal_mask .* 1f12) # size (T, T, B)
    wts = softmax(wts, dims=1)
    V ⊠ wts
end


"""
    MyMultiHeadAttention
"""

struct MyMultiHeadAttention{Q, K, V} <: LuxCore.AbstractLuxContainerLayer{(:query, :key, :value)}
    n_heads::Integer
    query::Q
    key::K
    value::V
end

# Q, K, V are calculated for all heads at once. Do do this, they have to map from (in_dim) => (n_heads * head_size)
function MyMultiHeadAttention(in_dim::Int, num_heads::Int; init_weight=glorot_uniform)
    mod(in_dim, num_heads) != 0 && DimensionMismatch("Number of heads must divide input dimension. Got: $(mod(in_dim, num_heads))")
    head_size = in_dim ÷ num_heads
    return MyMultiHeadAttention(
        num_heads,
        Dense(in_dim => num_heads * head_size; use_bias=false, init_weight=init_weight),
        Dense(in_dim => num_heads * head_size; use_bias=false, init_weight=init_weight),
        Dense(in_dim => num_heads * head_size; use_bias=false, init_weight=init_weight),
    )
end

LuxCore.initialstates(::AbstractRNG, ::MyMultiHeadAttention) = (query=NamedTuple(;), key=NamedTuple(;), value=NamedTuple(;))
    

function (mha::MyMultiHeadAttention)(x::AbstractArray, ps, st::NamedTuple)
    in_dim = size(ps.query.weight)[2]
    
    head_size = in_dim ÷ mha.n_heads
    # 1. Calculate Q, K, V
    q, st_q = mha.query(x, ps.query, st.query)
    k, st_k = mha.query(x, ps.query, st.query)
    v, st_v = mha.query(x, ps.query, st.query)

    #@show size(q), size(k), size(v)

    # 2. Reshape to split into different heads
    q_rs = reshape(q, (head_size, mha.n_heads, size(q)[2:end]...))
    k_rs = reshape(k, (head_size, mha.n_heads, size(q)[2:end]...))
    v_rs = reshape(v, (head_size, mha.n_heads, size(q)[2:end]...))

    #@show size(q_rs), size(k_rs), size(v_rs)

    # 3. Calculate attention scores for each head
    attn_scores = [calc_scaled_dot_prod_attn(q_rs[:, i, :, :], k_rs[:, i, :, :], v_rs[:, i, :, :]) for i in axes(q_rs, 2)]
    # 4. Concatenate the results
    return cat(attn_scores..., dims=1), (query=st_q, key=st_q, value=st_v)
end


"""
    Feed Forward 
"""
struct FeedForward{D1, D2, DO} <: LuxCore.AbstractLuxContainerLayer{(:dense_1, :dense_2, :dropout)}
    dense_1::D1
    dense_2::D2
    dropout::DO
end

function FeedForward(n_embd, p_dropout)
    return FeedForward(
        Dense(n_embd => 4 * n_embd, relu),
        Dense(4 * n_embd, n_embd),
        Dropout(p_dropout)
    )
end

function LuxCore.initialstates(rng::AbstractRNG, f::FeedForward) 
    (dense_1 = NamedTuple(;), dense_2 = NamedTuple(;), dropout=LuxCore.initialstates(rng, f.dropout))
end

function(ffwd::FeedForward)(x::AbstractArray, ps, st::NamedTuple)
    x, st_d1 = ffwd.dense_1(x, ps.dense_1, st.dense_1)
    x, st_d2 = ffwd.dense_2(x, ps.dense_2, st.dense_2)
    x, st_dr = ffwd.dropout(x, ps.dropout, st.dropout)
    x, (dense_1 = st_d1, dense_2 = st_d2, dropout = st_dr)
end


"""
    Transformer block

The Transformer block ties it all together: Multi-Head Attention, Feed-Forward, and Layer Norms.
"""
struct Transformer{H, F, L1, L2} <: LuxCore.AbstractLuxContainerLayer{(:mha, :ffwd, :ln1, :ln2)}
    mha::H      # Multi-Head Attention block
    ffwd::F
    ln1::L1
    ln2::L2
end

function Transformer(n_embd, n_head, seq_length)
    head_size = n_embd ÷ n_head
    Transformer(
        MyMultiHeadAttention(n_embd, head_size),
        FeedForward(n_embd, 0.1),
        LayerNorm((n_embd, seq_length)),
        LayerNorm((n_embd, seq_length))
    )
end

function LuxCore.initialstates(rng::AbstractRNG, t::Transformer)
    return (
        mha = Lux.initialstates(rng, t.mha),
        ffwd = Lux.initialstates(rng, t.ffwd),
        ln1 = Lux.initialstates(rng, t.ln1),
        ln2 = Lux.initialstates(rng, t.ln2)
    )
end

"""
Forward pass for the Transformer Block.

In simple terms, its:
x = x + mha(ln1(x))
x = x + ffwd(ln2(x))
"""
function(t::Transformer)(x::AbstractArray, ps, st::NamedTuple)
    x_ln1, st_ln1 = t.ln1(x, ps.ln1, st.ln1)
    x_mha, st_mha = t.mha(x_ln1, ps.mha, st.mha)
    
    x = x + x_mha

    x_ln2, st_ln2 = t.ln2(x, ps.ln2, st.ln2)
    x_ffwd, st_ffwd = t.ffwd(x_ln2, ps.ffwd, st.ffwd)

    return x + x_ffwd, (ln1 = st_ln1, ln2 = st_ln2, mha = st_mha, ffwd=st_ffwd)
end



