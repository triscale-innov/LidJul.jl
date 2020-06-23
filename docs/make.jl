push!(LOAD_PATH,"../src/")
using Documenter, LidJul

makedocs(
    modules = [LidJul],
    format = Documenter.HTML(prettyurls = false),
    checkdocs = :exports,
    sitename = "LidJul.jl",
    pages = Any["index.md"]
)

# deploydocs(
#     repo = "github.com/{GHUSER}/LidJul.jl.git",
# )
