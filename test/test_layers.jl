


@testset "SHA_layer" begin
    C, T, B = 32, 8, 4  # Channels, time, batch
    head_size = 16

    rng = Random.default_rng()
    Random.seed!(42)
    sha = SingleHeadAttention(C, head_size)
    ps, st = LuxCore.setup(rng, sha)

    # Test forward pass
    x = randn(Float32, C, T, B)


    x_out, st_new = sha(x, ps, st)

    @show st

    @test keys(st_new) == keys(st)
    @test size(x_out) == (head_size, T, B) 
    
end


@testset "MHA_layer" begin
    n_embd, T, B = 64, 16, 4 
    head_size = 16

    rng = Random.default_rng()
    Random.seed!(1337)
    mha = MyMultiHeadAttention(n_embd, head_size)
    ps, st = Lux.setup(rng, mha)

    # Test forward pass
    x = randn(Float32, n_embd, T, B)
    x_out, st_new = mha(x, ps, st)

    @test size(x_out) == size(x)
end

@testset "FeedForward" begin
   
    n_embd, T, B = 64, 16, 4

    rng = Random.default_rng()
    Random.seed!(1337)
    ffwd = FeedForward(n_embd, 0.1)

    ps, st = Lux.setup(rng, ffwd)

    # Test forward pass
    x = randn(Float32, n_embd, T, B)
    x_out, st_new = ffwd(x, ps, st)

    @test size(x_out) == size(x)

end

@testset "LayerNorm" begin
    n_embd, T, B = 64, 16, 4

    rng = Random.default_rng()
    ln = LayerNorm((n_embd, T))
    ps, st = Lux.setup(rng, ln)

    # Test forward pass
    x = 2.0 .* randn(Float32, (n_embd,T,B)) .+ 3.0;

    x_out, st_ = ln(x, ps, st)

    mean(x[:,:,1])
    mean(x_out[:,:,1])

    std(x[:,:,1])
    std(x_out[:,:,1])
end

@testset "TransformerBlock" begin
    n_embd, T, B = 64, 16, 3

    rng = Random.default_rng()
    t = Transformer(n_embd, 4, 16)

    ps, st = Lux.setup(rng, t)


end


