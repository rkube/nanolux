
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



function LuxCore.initialstates(::AbstractRNG, sha::SingleHeadAttention{Q, K, V}) where {Q, K, V} 
    mask = tril(fill(true, sha.max_sequence_length, sha.max_sequence_length), -1)
    return (causal_mask = mask, query=NamedTuple(;), key = NamedTuple(;), value=NamedTuple(;))
end


function (sha::SingleHeadAttention)(x::AbstractArray, ps, st::NamedTuple)
    # x has shape (features, sequence_length, batch_size)
    xT = eltype(x)
    T = size(x, 2)
    head_size = size(ps.query.weight, 1)

    T > sha.max_sequence_length && error("Input sequence length $(T) exceeds max sequence length $(sha.max_sequence_length)")

    q, st_q = sha.query(x, ps.query, st.query)  # size = (head_size, T, B)
    k, st_k = sha.key(x, ps.key, st.key)        # size = (head_size, T, B)
    v, st_v = sha.value(x, ps.value, st.value)  # size = (head_size, T, B) 


    #wts = permutedims(q, (2, 1, 3)) ⊠ k ./ xT(sqrt(head_size)) # size = (T,T)
    wts = permutedims(k, (2, 1, 3)) ⊠ q ./ xT(sqrt(head_size)) # size = (T,T)
    # We later apply softmax. For large values, softmax may converge to a one-hot vector.
    # If we scale them, the resulting distribution after softmax will be more diffuse
    causal_mask = view(st.causal_mask, 1:T, 1:T)
    wts = wts .- (causal_mask .* 1f12) # size (T, T, B)
    # Normalize to get a probability
    wts = softmax(wts, dims=1)
    v ⊠ wts, (causal_mask = st.causal_mask, query = st_q, key = st_k, value = st_v)  # Size (C, T, B)
end


