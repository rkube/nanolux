


@testset "SHA_layer" begin
    rng = Random.default_rng()
    Random.seed!(42)
    sha = SingleHeadAttention(16, 4)
    ps, st = LuxCore.setup(rng, sha)

    # Test forward pass
    x = randn(16,8,4)


    x_out, st_new = sha(x, ps, st)

    @show st
    
end
