using BenchmarkTools
using LinearAlgebra
using NNlib
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

# A function that calculates the x[t,b] = mean_{i<=t} x[b,i] at 
# https://youtu.be/kCc8FmEb1nY?t=2759
# Trying to be nice with memory allocation
# Note: In torch, you can omit indices in a way you can't in Julia
# For example, in python x[0,0] means the same as x[0,0,:].
# You can index x with only 2 indices although it is 3 dimensional.
# Julia requires you to write out the third dimension
function mean_thing_view(x)

    xbow = zeros(eltype(x), C, T, B) 
    for ix_b in 1:B
        for ix_t in 1:T
            xprev = view(x, :, 1:ix_t, ix_b)
            m = mean(xprev, dims=2)
            xbow[:, ix_t, ix_b] .= m[:, 1]
            #@show xprev
        end
    end
    xbow
end


@benchmark mean_thing_view(x)
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



function mean_thing_noview(x)

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


@benchmark mean_thing_noview(x)


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

# The trick in self-attention
a = ones(3, 3)
b = [2.0 7.0; 6.0 4.0; 6.0 5.0]
c = a * b

# To create triangular matrces in Julia, we call the tril function, just as in torch.
a = tril(ones(3,3))
b = [2.0 7.0; 6.0 4.0; 6.0 5.0]
c = a * b

# We can adapt this to calculate a mean from a sum
a = tril(ones(3,3)) 
a = a ./ sum(a, dims=2) # Divide by the row-wise sum. We don't need keepdim=True, as in pytorch.
b = [2.0 7.0; 6.0 4.0; 6.0 5.0]
# Now the rows of c store the average of all the elements deposited in the row.
c = a * b

# The central idea is that we can formulate the mean_{i≤t} operation by using a lower triangular matrix.
# In particular
#
wts = triu(ones(T, T))
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
xbow3 ≈ x ⊠ reshape(wts, (8,8,1)) 
