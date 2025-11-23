
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



function estimate_loss(model, ps, st, data_loader; max_iter=1000)
    println("Estimating loss")
    loss_total = 0.0
    loss_fn = CrossEntropyLoss(; agg=mean, logits=Val(true))

    # We are working with reactant. That means, we have to do some precompiling of the functions
    # that are called with reactant arrays. Remember, ps, xb, and yb are now all reactant arrays! 
    (xb, yb) = first(data_loader)
    onehotbatch_c = @compile onehotbatch(yb, 1:65)
    model_c = @compile model(xb, ps, st)
    

    for (ix_b, batch) in ProgressBar(enumerate(data_loader))
        ix_b > max_iter && break
        xb, yb = batch
        yb_hot = onehotbatch_c(yb, 1:65)
        y_pred, _ = model_c(xb, ps, st)
        loss_total += loss_fn(y_pred, yb_hot)
    end
    return loss_total / max_iter
end

function train(tstate::Training.TrainState, vjp, loader_train, loader_valid, num_iter)
    println("Training")

    loss_iter = 0.0

    time_start = time_ns()

    (xb, yb) = first(loader_train)
    onehotbatch_c = @compile onehotbatch(yb, 1:65)

    for (ix_it, batch) in enumerate(loader_train)
        ix_it > num_iter && break
        xb, yb = batch
        yb_hot = onehotbatch_c(yb, 1:65)
        _, loss, _, tstate = Training.single_train_step!(vjp, CrossEntropyLoss(; agg=mean, logits=Val(true)), (xb, yb_hot), tstate)
        loss_iter += loss

        if mod(ix_it, 100) == 0
            time_stop = time_ns()
            ps_c = tstate.parameters #|> cdev
            st_c = Lux.testmode(tstate.states) #|> cdev
            loss_train = estimate_loss(tstate.model, ps_c, st_c, loader_train)
            loss_valid = estimate_loss(tstate.model, ps_c, st_c, loader_valid)
            time_elapsed_s = (time_stop - time_start) / 1e9
            println("Iteration $(ix_it). Training loss: $(loss_iter / num_iter). Validation loss: $(loss_valid). Elapsed time: $(round(time_elapsed_s, digits=4))s")
#            println("Elapsed time: $(round(time_elapsed_s, digits=4))s")
            time_start = time_ns()
        end
    end
    loss_total = loss_iter / num_iter
    println("----- Final loss: $(loss_total)")
    return tstate
end


function get_model(vocab_size, n_embd, num_heads, head_size)
    num_heads = n_embd ÷ head_size
    model = @compact(token_embedding = Embedding(vocab_size => n_embd),
        pos_embedding = Embedding(vocab_size => n_embd),
        trf_block_1 = Transformer(n_embd, head_size),
        trf_block_2 = Transformer(n_embd, head_size),
        trf_block_3 = Transformer(n_embd, head_size),
        trf_block_4 = Transformer(n_embd, head_size),
        lm_head = Dense(n_embd => vocab_size)
        ) do x
    T, B = size(x)     # T: block_size (the sequence length), B: batch_size
    tok_emb = token_embedding(x)   # size (C, T, B)
    pos_emb = pos_embedding(collect(1:T))   # size (C, T)
    x = tok_emb .+ pos_emb
    x = trf_block_1(x)
    x = trf_block_2(x)
    x = trf_block_3(x)
    x = trf_block_4(x)
    logits = lm_head(x)
    @return logits
    end
    return model
end


function runme()
    rng = Random.default_rng()
    Random.seed!(rng, 1337)
    n_embd = 64
    batch_size = 16
    block_size = 32
    head_size = 16
    num_heads = n_embd ÷ head_size

    xdev = reactant_device()
    cdev = cpu_device()

    d = NanoDataset(DATAFILE, block_size)
    d_train, d_valid = splitobs(d, at=0.8)
    loader_train = DataLoader(d_train, batchsize=batch_size, collate=true, shuffle=true, partial=false)
    loader_valid = DataLoader(d_valid, batchsize=batch_size, collate=true, shuffle=true, partial=false)

    vocab_size = get_vocab_size(d)

    model = get_model(vocab_size, n_embd, num_heads, head_size)
    ps, st = Lux.setup(rng, model)

    #x = rand(1:65, block_size, batch_size)
    #y_p, _ = model(x, ps, st)

    # Move things on to the device
    ps_ra = ps |> xdev 
    st_ra = st |> xdev
    loader_train = loader_train |> xdev 
    loader_valid = loader_valid |> xdev

    opt = Adam(1f-3)

    # Create an initial training state
    tstate = Training.TrainState(model, ps_ra, st_ra, opt)
    # Save the trained parameters of the model
    tstate_fin = train(tstate, AutoEnzyme(), loader_train, loader_valid, 5_000) 
    println("Training done")

    # Generate output of the trained model
    ps_train = tstate_fin.parameters |> cdev
    xb, _ = first(loader_valid) |> cdev

    model_output = generate(xb, 200, model, ps_train, st);

    println("Model output:")
    print(join(decode(d, model_output[:, 2]), ""))
end


runme()

