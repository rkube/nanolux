
export SingleHeadAttention

"""
    SingleHeadAttention

This stores the three dense layers needed to calculate single head attention
"""
struct SingleHeadAttention{Q, K, V} <: LuxCore.AbstractLuxContainerLayer{(:query, :key, :value)}
    query::Q 
    key::K 
    value::V
    max_sequence_length::Int
end

function SingleHeadAttention(in_dim::Int, head_size::Int; init_weight=glorot_uniform) 
    return SingleHeadAttention(
        Dense(in_dim => head_size; use_bias=false, init_weight=init_weight),
        Dense(in_dim => head_size; use_bias=false, init_weight=init_weight),
        Dense(in_dim => head_size; use_bias=false, init_weight=init_weight),
        128
    )
end


#function LuxCore.initialparameters(rng::AbstractRNG, sha::SingleHeadAttention)
#    return (query.weight = sha.init_weight(rng, sha.head_size, sha.vocab_size),
#            key.weight = sha.init_weight(rng, sha.head_size, sha.vocab_size),
#            value.weight = sha.init_weight(rng, sha.head_size, sha.vocab_size))
#end

function LuxCore.initialstates(::AbstractRNG, sha::SingleHeadAttention{Q, K, V}) where {Q, K, V} 
    mask = tril(fill(true, sha.max_sequence_length, sha.max_sequence_length), -1)
    return (causal_mask = mask, )
end


function (sha::SingleHeadAttention)(x::AbstractArray, ps, st::NamedTuple)
    # x has shape (features, sequence_length, batch_size)
    T = size(x, 3)
    head_size = size(ps.query.weight, 1)

    T > sha.max_sequence_length && error("Input sequence length $(T) exceeds max sequence length $(sha.max_sequence_length)")

    # Calculate query, key, and value vectors
    q, st_q = sha.query(x, ps.query, st.query)
    k, st_k = sha.key(x, ps.key, st.key)
    v, st_v = sha.value(x, ps.value, st.value)


    wts = permutedims(q, (2, 1, 3)) ⊠ k # size = (T,T)
    causal_mask = view(st.causal_mask, 1:T, 1:T)
    wts = wts .- (causal_mask .* 1e12)
   # Normalize so we get a probability
    wts = softmax(wts, dims=1)
    # Instead of working with token embeddings directly, work with the value
    v ⊠ wts, (query = st_q, key = st_k, value = st_v)
end


