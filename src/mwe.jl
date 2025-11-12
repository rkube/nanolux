using Lux
using Reactant
using Random

model = @compact(token_embedding = Embedding(65 => 32)) do x 
          tok_embd = token_embedding(x)
          @return tok_embd
        end

Reactant.set_default_backend("cpu")
rng = Random.default_rng()
Random.seed!(42)
xdev = reactant_device()
cdev = cpu_device()

x = rand(1:65, 16, 4) 

ps, st = Lux.setup(rng, model)
model(x, ps, st)


ps_ra = ps |> xdev
st_ra = st |> xdev

x_ra = rand(1:65, 16, 4)  |> xdev

model_compiled = @compile model(x_ra, ps_ra, Lux.testmode(st_ra))

model_compiled(x_ra, ps_ra, st_ra)




