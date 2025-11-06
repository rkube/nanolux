


@testset "SHA_layer" begin
    rng = Random.default_rng()
    Random.seed!(42)
    sha = SingleHeadAttention(16, 4)
    ps, st = LuxCore.setup(rng, sha)

    
end
