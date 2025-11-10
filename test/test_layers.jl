


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
