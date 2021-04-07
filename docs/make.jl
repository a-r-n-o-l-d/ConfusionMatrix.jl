using ConfusionMatrix
using Documenter

DocMeta.setdocmeta!(ConfusionMatrix, :DocTestSetup, :(using ConfusionMatrix); recursive=true)

makedocs(;
    modules=[ConfusionMatrix],
    authors="Arnold",
    repo="https://github.com/a-r-n-o-l-d/ConfusionMatrix.jl/blob/{commit}{path}#{line}",
    sitename="ConfusionMatrix.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://a-r-n-o-l-d.github.io/ConfusionMatrix.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/a-r-n-o-l-d/ConfusionMatrix.jl",
)
