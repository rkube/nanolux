
export SingleHeadAttention

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



function LuxCore.initialstates(::AbstractRNG, sha::SingleHeadAttention{Q, K, V}) where {Q, K, V} 
    return (query=NamedTuple(;), key = NamedTuple(;), value=NamedTuple(;))
end


function (sha::SingleHeadAttention)(x::AbstractArray, ps, st::NamedTuple)
    # x has shape (features, sequence_length, batch_size)
    xT = eltype(x)
    T = size(x, 2)
    head_size = size(ps.query.weight, 1)

    q, st_q = sha.query(x, ps.query, st.query)  # size = (head_size, T, B)
    k, st_k = sha.key(x, ps.key, st.key)        # size = (head_size, T, B)
    v, st_v = sha.value(x, ps.value, st.value)  # size = (head_size, T, B) 

    wts = permutedims(k, (2, 1, 3)) ⊠ q ./ xT(sqrt(head_size)) # size = (T,T)

    #         <k1, q1>  <k1, q2>  ... <k1, qT>
    #         <k2, q1>  <k2, q2>  ... <k2, qT>
    #            ...      ...     ...   ...
    #  wts =  <kT, q1>  <kT, q2>  ...  <kT, qT>
    #
    # The mask needs to prevent a token at a given position to attending to subsequent tokens.
    # That is, <k_i, q_j> for i > j. These scalar products are in the lower triangular, below
    # the main diagonal. A lower triangular matrix from diagonal -1, does the trick.

    # We later apply softmax. For large values, softmax may converge to a one-hot vector.
    # If we scale them, the resulting distribution after softmax will be more diffuse
    causal_mask = tril(fill(true, T, T), -1)

    wts = wts .- (causal_mask .* 1f12) # size (T, T, B)
    # Normalize to get a probability
    wts = softmax(wts, dims=1)
    v ⊠ wts, (query = st_q, key = st_k, value = st_v)  # Size (C, T, B)
    

end


#"""
#    MultiHeadAttention
#"""
#
#struct MultiHeadAttention
#    heads
#    proj
#end
#
#function MultiHeadAttention(n_embd, num_heads)
#    head_size = n_embd ÷ num_heads
#    return MultiHeadAttention(
#        [SingleHeadAttention(n_embd, head_size) for _ in 1:num_heads],
#        Dense(n_embd => n_embd, relu)
#    )
#end
#
#function LuxCore.initialstates(rng::AbstractRNG, mha::MultiHeadAttention)
#    return(heads=[LuxCore.initialstates(rng) for _ in 1:num_heads], proj=Luxcore.initialstates(mha.proj))
#end
#
#function (mha::MultiHeadAttention)(x::AbstractArray, ps, st::NamedTuple)
#    num_heads = length(mha.heads)
#    out = Parallel(vcat, [mha.heads[i](x, ps, st) for i in num_heads]...)
#    out = mha.proj(out, ps.dense, st.dense)
#    out
#end
#

