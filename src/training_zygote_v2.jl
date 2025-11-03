
using ADTypes
using BenchmarkTools
using Distributions
using Lux
using MLUtils
using OneHotArrays
using Optimisers
using Random
using Zygote

using NanoLux


# Instead of the loss_fun we previously defined, we can simplify quiet a bit: 
# 1. Lux CrossEntropyLoss takes automatically care of reshaping from
#    (C, T, B) -> (C, T*B)
# 2. Converting yb into a one-hot batch is done in the main training loop.
#
# So we don't really need to have a custom-defined loss function that adheres to the TrainingState API.
# Instead, we just pass the CrossEntropyLoss(; agg=mean, logits=Val(True)) into the single_train_step! function.



function train(tstate::Training.TrainState, vjp, data_loader, num_epochs)
    for epoch in 1:num_epochs
        loss_epoch = 0.0
        for item in data_loader
            xb, yb = item
            yb_hot = onehotbatch(yb, 1:65)
            _, loss, _, tstate = Training.single_train_step!(vjp, CrossEntropyLoss(; agg=mean, logits=Val(true)), (xb, yb_hot), tstate)
            loss_epoch += loss
        end
        loss_epoch /= length(data_loader)

        if mod(epoch, 1) == 0
            println("Epoch: $(epoch)    Loss: $(loss_epoch)")
        end
    end
    return tstate
end

function runme()
    rng = Random.default_rng()
    Random.seed!(rng, 1337)
    batch_size = 4
    block_size = 8
    n_embed = 32

    d = NanoDataset(DATAFILE, block_size)
    vocab_size = get_vocab_size(d)

    data_loader = DataLoader(d, batchsize=batch_size, collate=true, shuffle=true)
    xb, yb = first(data_loader)


    model = Chain(Embedding(vocab_size => n_embed), Dense(n_embed => vocab_size))
    ps, st = Lux.setup(rng, model)

    model(xb, ps, st)

    opt = Adam(1f-3)

    # Create an initial training state
    tstate = Training.TrainState(model, ps, st, opt)
    # Save the trained parameters of the model
    tstate_fin =  train(tstate, AutoZygote(), data_loader, 1)

    # Generate output of the trained model
    ps_train = tstate_fin.parameters
    xb, _ = first(data_loader)

    model_output = generate(xb, 100, model, ps_train, st)

    println("Model output:")
    print(join(decode(d, model_output[:, 1]), ""))
end


runme()

