using Documenter
using Literate
using NanoLux

# Directory where the examples live
const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR = joinpath(@__DIR__, "src", "literated")


example_scripts = ["attention.jl", "dataloading.jl"]

for n in 1:length(example_scripts)
    example_filepath = joinpath(EXAMPLES_DIR, example_scripts[n])
    Literate.markdown(example_filepath, OUTPUT_DIR)
end

example_pages = ["Attention" => "literated/attention.md", "Dataloading" => "literated/dataloading.md"]


makedocs(sitename="NanoLux documentation", remotes=nothing,
         pages = ["Home" => "index.md",
                  "Examples" => example_pages
                ]
         )
