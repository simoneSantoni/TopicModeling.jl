using TopicModeling
using Documenter

DocMeta.setdocmeta!(TopicModeling, :DocTestSetup, :(using TopicModeling); recursive=true)

makedocs(;
    modules=[TopicModeling],
    checkdocs=:exports,
    authors="Simone Santoni",
    repo="https://github.com/simoneSantoni/TopicModeling.jl/blob/{commit}{path}#{line}",
    sitename="TopicModeling.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://simoneSantoni.github.io/TopicModeling.jl",
        repolink="https://github.com/simoneSantoni/TopicModeling.jl",
        assets=["assets/favicon.ico"],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Models" => "models.md",
        "Performance" => "performance.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/simoneSantoni/TopicModeling.jl",
    devbranch="main",
)
