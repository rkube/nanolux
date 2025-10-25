# Differences and similarities betwee Lux.jl and Pytorch
using Lux
using OneHotArrays
#
#
# Softmax is the same in both languages
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# torch.manual_seed(1337)
# input = torch.randn(3, 5).transpose(1, 0)
#
#tensor([[0.0900, 0.2447, 0.6652],
#        [0.0900, 0.2447, 0.6652],
#        [0.0900, 0.2447, 0.6652],
#        [0.0900, 0.2447, 0.6652],
#        [0.0900, 0.2447, 0.6652]])
# target = torch.randns(3, 5).softmax(dim=1)

# While Lux gives
softmax(reshape(-7:7, 3, 5), dims=1)
```julia
3×5 Matrix{Float64}:
 0.0900306  0.0900306  0.0900306  0.0900306  0.0900306
 0.244728   0.244728   0.244728   0.244728   0.244728
 0.665241   0.665241   0.665241   0.665241   0.665241

# Note that the output is the transposed of the torch array. Even though we pass dim=1 to the 
# calls to softmax in julia and python, they operate over different dimensions since python
# starts indexing at 0 while dims=1 is the first dimension in Julia.
#
# 
#

"""python
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

input = torch.randn(5, 3)
print(input)



tensor([[-2.0260, -2.0655, -1.2054],
        [-0.9122, -1.2502,  0.8032],
        [-0.2071,  0.0544,  0.1378],
        [-0.3889,  0.5133,  0.3319],
        [ 0.6300,  0.5815, -0.0282]])


F.cross_entropy(input, torch.tensor([0, 1, 2, 0, 1]))
tensor(1.4786)

"""


# When we define the same tensors in Julia, we have to do so in reverse dimension order
input = [-2.0260 -0.9122 -0.2071 -0.3889  0.6300;
         -2.0655 -1.2502  0.0544  0.5133  0.5815;
         -1.2054  0.8032  0.1378  0.3319 -0.0282]
oh = onehotbatch([1, 2, 3, 1, 2], 1:3)
CrossEntropyLoss(; agg=mean, logits=true)(input, oh)
# This yields the same result as in pytorch

```julia-repl
julia> CrossEntropyLoss(; agg=mean, logits=true)(input, oh)
1.4785573763907676
```

