

export generate, sample_categorical_batch

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
        #@show ix_t
        # 1. Get the logits from the model.
        #    Here we pass only the last token,
        #@show size(current_token)
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
        tokens_out[T+ix_t, :] .= idx_next[1,:] 

        # 6. Update the context window for the next iteration 
        #    This context window is a sliding window of shape T.
        #    It needs to be this size because the size of the positional embedding is T.
        current_token = view(tokens_out, 1+ix_t:T+ix_t, :)
    end
    return tokens_out
end


function generate_v2(input, max_new_tokens, model, ps, st)
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
        #@show ix_t
        # 1. Get the logits from the model.
        #    Here we pass only the last token,
        #@show size(current_token)
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
        tokens_out[T+ix_t, :] .= idx_next[1,:]

        # 6. Update the context window for the next iteration
        #    This context window is a sliding window of shape T.
        #    It needs to be this size because the size of the positional embedding is T.
        current_token = view(tokens_out, 1+ix_t:T+ix_t, :)
    end
    return tokens_out
end
