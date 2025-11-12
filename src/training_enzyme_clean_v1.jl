
using ADTypes
using BenchmarkTools
using Enzyme
using Lux
using MLUtils
using OneHotArrays
using Optimisers
using Random
using Reactant
using ProgressBars

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
    cdev = cpu_device()
    xdev = reactant_device()
    println("Training")

    for epoch in 1:num_epochs
        loss_epoch = 0.0
        for (ix_batch, batch) in enumerate(loader_train)
            xb, yb = batch
            batch_size = last(size(xb))
            yb_hot = onehotbatch(cdev(yb), 1:65) |> xdev
            _, loss, _, tstate = Training.single_train_step!(vjp, CrossEntropyLoss(; agg=mean, logits=Val(true)), (xb, yb_hot), tstate)
            loss_epoch += loss 

            # Calculate validation loss over 100 samples
            #if mod(ix_batch, 1000) == 0
            #    n_samples = 100
            #    # Evaluate loss on train and validation set
            #    loss_valid = 0.0
            #    st_ = Lux.testmode(tstate.states)
            #    for (ix_bv, batch_v) ∈ enumerate(loader_valid)
            #        ix_bv > n_samples && break
            #        xv_b, yv_b = batch_v
            #        yv_b_hot = onehotbatch(cdev(yv_b), 1:65) |> xdev
            #        xv_out, _ = tstate.model(xv_b, tstate.parameters, st_)
            #        loss_valid += CrossEntropyLoss(; agg=mean, logits=Val(true))(xv_out, yv_b_hot)
            #    end
            #    loss_valid /= n_samples
            #    println("Batch $(ix_batch)   Validation loss=$(loss_valid)")
            #    loss_valid = 0.0
            #end
        end
        loss_epoch = loss_epoch / length(loader_train) 

        if mod(epoch, 1) == 0
            println("Epoch: $(epoch)    Loss: $(loss_epoch)")
        end
    end
    return tstate
end


function estimate_loss(model, ps, st, data_loader; max_iter=1000)
    loss_total = 0.0
    loss_fn = CrossEntropyLoss(; agg=mean, logits=Val(true))
    for (ix_b, batch) in ProgressBar(enumerate(data_loader))
        ix_b > max_iter && break
        xb, yb = batch
        yb_hot = onehotbatch(cdev(yb), 1:65) |> xdev
        y_pred, _ = model(xb, ps, st)
        loss_total += loss_fn(y_pred, yb_hot)
    end
    return loss_total / max_iter
end

function train_v2(tstate::Training.TrainState, vjp, loader_train, loader_valid, num_iter)
    cdev = cpu_device()
    xdev = reactant_device()
    println("Training")

    loss_iter = 0.0

    for (ix_it, batch) in enumerate(loader_train)
        ix_it > num_iter && break
        xb, yb = batch
        yb_hot = onehotbatch(cdev(yb), 1:65) |> xdev
        _, loss, _, tstate = Training.single_train_step!(vjp, CrossEntropyLoss(; agg=mean, logits=Val(true)), (xb, yb_hot), tstate)
        loss_iter += loss

        if mod(ix_it, 100) == 0
            ps_c = tstate.parameters |> cdev
            st_c = Lux.testmode(tstate.states) |> cdev
            loss_train = estimate_loss(tstate.model, ps_c, st_c, loader_train)
            loss_valid = estimate_loss(tstate.model, ps_c, st_c, loader_valid)
            println("Iteration $(ix_it). Training loss: $(loss_train). Validation loss: $(loss_valid)")
        end

    end
    loss_total = loss_iter / num_iter
    println("----- Final loss: $(loss_total)")
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
    batch_size = 16
    block_size = 32
    head_size = 16
    n_embd = 64

    xdev = reactant_device()
    cdev = cpu_device()

    d = NanoDataset(DATAFILE, block_size)
    d_train, d_valid = splitobs(d, at=0.8)
    loader_train = DataLoader(d_train, batchsize=batch_size, collate=true, shuffle=true, partial=false)
    loader_valid = DataLoader(d_valid, batchsize=batch_size, collate=true, shuffle=true, partial=false)

    vocab_size = get_vocab_size(d)
    model = get_model(vocab_size, n_embd, head_size)
    ps, st = Lux.setup(rng, model)

    x = rand(1:65, 8, 4)
    model(x, ps, st)

    # Move things on to the device
    ps_ra = ps |> xdev 
    st_ra = st |> xdev
    loader_train = loader_train |> xdev 
    loader_valid = loader_valid |> xdev

    opt = Adam(1f-3)

    # Create an initial training state
    tstate = Training.TrainState(model, ps_ra, st_ra, opt)
    # Save the trained parameters of the model
    #tstate_fin = train(tstate, AutoEnzyme(), loader_train, loader_valid, 1) 
    tstate_fin = train_v2(tstate, AutoEnzyme(), loader_train, loader_valid, 500) 
    println("Training done")

    # Generate output of the trained model
    ps_train = tstate_fin.parameters |> cdev
    xb, _ = first(loader_valid) |> cdev

    model_output = generate_v2(xb, 50, model, ps_train, st)

    println("Model output:")
    print(join(decode(d, model_output[:, 9]), ""))
end


runme()

