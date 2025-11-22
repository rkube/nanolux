# I'm curious about how onehot batches are calculated and used in Enzyme
#
using Lux
using OneHotArrays
using Enzyme
using Random
using Reactant
using Statistics


# Generate some test data with 65 categories
rng = Random.default_rng()
Random.seed!(1337)
x = softmax(randn(7, 16), dims=1)

y = rand(1:7, 16)
y_oh = onehotbatch(y, 1:7)

# This works fine
CrossEntropyLoss(; agg=mean, logits=Val(false))(x, y_oh)

# Now use reactant devices 
xdev = reactant_device()

x_r = x |> xdev
y_r = y |> xdev

# remember to use @jit
y_oh = @jit  onehotbatch(y_r, 1:7)

# This will fail
y_oh = onehotbatch(y_r, 1:7)


