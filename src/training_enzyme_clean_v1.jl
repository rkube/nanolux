
using ADTypes
using BenchmarkTools
using Enzyme
using Lux
using MLUtils
using OneHotArrays
using Optimisers
using Random
using Reactant

using NanoLux


# Instead of the loss_fun we previously defined, we can simplify quiet a bit: 
# 1. Lux CrossEntropyLoss takes automatically care of reshaping from
#    (C, T, B) -> (C, T*B)
# 2. Converting yb into a one-hot batch is done in the main training loop.
#
# So we don't really need to have a custom-defined loss function that adheres to the TrainingState API.
# Instead, we just pass the CrossEntropyLoss(; agg=mean, logits=Val(True)) into the single_train_step! function.



function train(tstate::Training.TrainState, vjp, loader_train, loader_valid, num_epochs) 
    # Move things to cpu since we can't have onehotbatches for ConcretePJRTArrays
    for epoch in 1:num_epochs
        loss_epoch = 0.0
        for (ix_batch, batch) in enumerate(loader_train)
            xb, yb = batch
            yb_hot = onehotbatch(yb, 1:65)
            _, loss, _, tstate = Training.single_train_step!(vjp, CrossEntropyLoss(; agg=mean, logits=Val(true)), (xb, yb_hot), tstate)
            loss_epoch += loss

            if mod(ix_batch, 100) == 0
                # Evaluate loss on train and validation set
                loss_valid = 0.0
                for (ix_b, batch) ∈ enumerate(loader_valid)
#                    #xb, yb = batch
#                    #xb |> cdev
#                    #yb |> cdev
#                    #yb_hot = onehotbatch(yb, 1:65) |> xdev
                    loss_valid += 0.1
#                    #x_out, _ = tstate.model(xb, ps, st)
#                    #loss_train += CrossEntropyLoss(; agg=mean, logits=Val(true))(x_out, yb_hot)
                end
                println("Batch $(ix_batch)   Validation loss=$(loss_valid)")
                loss_valid = 0.0
            end
        end
        loss_epoch /= length(data_loader)

        if mod(epoch, 1) == 0
            println("Epoch: $(epoch)    Loss: $(loss_epoch)")
        end
    end
    return tstate
end


function get_model(vocab_size, n_embd, head_size)
    model = @compact(token_embedding = Embedding(vocab_size => n_embd),
                     pos_embedding = Embedding(vocab_size => n_embd),
                     sa_head = SingleHeadAttention(n_embd, n_embd),
                     lm_head = Dense(n_embd => vocab_size)
                     ) do x
                     T, B = size(x)     # T: block_size (the sequence length), B: batch_size
                     tok_emb = token_embedding(x)   # size (C, T, B)
                     pos_emb = pos_embedding(1:T)   # size (C, T)
                     x = tok_emb .+ pos_emb
                     x = sa_head(x)
                      
                     logits = lm_head(x)
                     @return logits
                     end
    return model
end


function runme()
    rng = Random.default_rng()
    Random.seed!(rng, 1337)
    batch_size = 128
    block_size = 32
    head_size = 16
    n_embd = 64

    xdev = reactant_device()
    cdev = cpu_device()

    d = NanoDataset(DATAFILE, block_size)
    d_train, d_valid = splitobs(d, at=0.8)
    loader_train = DataLoader(d_train, batchsize=batch_size, collate=true, shuffle=true)
    loader_valid = DataLoader(d_valid, batchsize=batch_size, collate=true, shuffle=true)


    vocab_size = get_vocab_size(d)
    model = get_model(vocab_size, n_embd, head_size)
    ps, st = Lux.setup(rng, model)

    x = rand(1:65, 8, 4)
    model(x, ps, st)

    # Move things on to the device
    ps_ra = ps |> xdev 
    st_ra = st |> xdev
    loader_train |> xdev 
    loader_test |> xdev

    opt = Adam(1f-3)

    # Create an initial training state
    tstate = Training.TrainState(model, ps_ra, st_ra, opt)
    # Save the trained parameters of the model
    tstate_fin = train(tstate, AutoEnzyme(), loader_train, loader_valid, 1) 
    println("Training done")

    # Generate output of the trained model
    ps_train = tstate_fin.parameters |> cdev
    xb, _ = first(data_loader)

    model_output = generate(xb, 20, model, ps_train, st)

    println("Model output:")
    print(join(decode(d, model_output[:, 5]), ""))
end


runme()

