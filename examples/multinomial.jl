# Let's investigate multinomial sampling a bit. I couldn't find a Julia implementation.
#
# This is the behaviour I want to reproduce in Julia
# 
#
# import torch
# torch.manual_seed(1337)
# logits = torch.randn((4, 16))
# 
# tensor([[ 0.1808, -0.0700, -0.3596, -0.9152,  0.6258,  0.0255,  0.9545,  0.0643,
#           0.3612,  1.1679, -1.3499, -0.5102,  0.2360, -0.2398, -0.9211,  1.5433],
#         [ 1.3488, -0.1396,  0.2858,  0.9651, -2.0371,  0.4931,  1.4870,  0.5910,
#           0.1260, -1.5627, -1.1601, -0.3348,  0.4478, -0.8016,  1.5236,  2.5086],
#         [-0.6631, -0.2513,  1.0101,  0.1215,  0.1584,  1.1340, -1.1539, -0.2984,
#          -0.5075, -0.9239,  0.5467, -1.4948, -1.2057,  0.5718, -0.5974, -0.6937],
#         [ 1.6455, -0.8030,  1.3514, -0.2759, -1.5108,  2.1048,  2.7630, -1.7465,
#           1.4516, -1.5103,  0.8212, -0.2115,  0.7789,  1.5333,  1.6097, -0.4032]])
#
#
#
# probs = torch.softmax(logits, dim=1) # (B, C)
# print(probs)
# tensor([[0.0534, 0.0416, 0.0311, 0.0179, 0.0834, 0.0457, 0.1158, 0.0475, 0.0640,
#          0.1433, 0.0116, 0.0268, 0.0564, 0.0351, 0.0177, 0.2087],
#         [0.1016, 0.0229, 0.0351, 0.0692, 0.0034, 0.0432, 0.1166, 0.0476, 0.0299,
#          0.0055, 0.0083, 0.0189, 0.0412, 0.0118, 0.1210, 0.3239],
#         [0.0311, 0.0469, 0.1656, 0.0681, 0.0707, 0.1875, 0.0190, 0.0448, 0.0363,
#          0.0239, 0.1042, 0.0135, 0.0181, 0.1069, 0.0332, 0.0301],
#         [0.0947, 0.0082, 0.0705, 0.0139, 0.0040, 0.1499, 0.2894, 0.0032, 0.0780,
#          0.0040, 0.0415, 0.0148, 0.0398, 0.0846, 0.0913, 0.0122]])
#
# probs.sum(dims=1)
# tensor([1.0000, 1.0000, 1.0000, 1.0000])
# idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
# print(idx_next)
# tensor([[12],
#         [ 0],
#         [ 4],
#         [14]])



# Here is some code that gemini spit out

using Distributions
using GLMakie
using Lux 


#
# In pytorch, the BigramLanguageModel samples from a multinomial distribution in the
# generate method. How do we do something similar in Julia?

"""
Batched categorical sampling.
`probs` is a (Classes, Batch) matrix.
Returns a (1, Batch) matrix of sampled indices.
"""
function sample_categorical_batch(probs::AbstractMatrix)
    B, C = size(probs, 1)
    # Pre-allocate the output matrix to get next token for each batch
    idx_next = Matrix{Int}(undef, B, 1)

    # `torch.multinomial` expects probabilities, but is robust to
    # small floating point errors not summing to 1.
    # `Categorical` is stricter, so we normalize each column.
    probs_normalized = probs ./ sum(probs; dims=2)
    
    # Loop over each item in the batch (each row)
    for i in 1:B
        # Create a distribution from the probabilities for this batch item
        dist = Distributions.Categorical(probs_normalized[i, :])
        # Sample one index (e.g., 1, 2, or 3... not 0-based)
        idx_next[i, 1] = rand(dist)
    end
    
    return idx_next
end



logits = [ 0.1808 -0.0700 -0.3596 -0.9152  0.6258  0.0255  0.9545  0.0643 0.3612  1.1679 -1.3499 -0.5102  0.2360 -0.2398 -0.9211  1.5433;
           1.3488 -0.1396  0.2858  0.9651 -2.0371  0.4931  1.4870  0.5910 0.1260 -1.5627 -1.1601 -0.3348  0.4478 -0.8016  1.5236  2.5086;
          -0.6631 -0.2513  1.0101  0.1215  0.1584  1.1340 -1.1539 -0.2984 -0.5075 -0.9239  0.5467 -1.4948 -1.2057  0.5718 -0.5974 -0.6937;
           1.6455 -0.8030  1.3514 -0.2759 -1.5108  2.1048  2.7630 -1.7465 1.4516 -1.5103  0.8212 -0.2115  0.7789  1.5333  1.6097 -0.4032]

probs = softmax(logits, dims=2)

# This gives a reasonable plot comparing to the PDF plotted from pytorch
hist(rand(Distributions.Categorical(probs[1, :]), 100))

ix_next = sample_categorical_batch(probs)




