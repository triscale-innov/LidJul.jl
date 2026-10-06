using Documenter, LidJul

DocMeta.setdocmeta!(LidJul,:DocTestSetup,:(using LidJul);recursive=true)
makedocs(
    modules=[LidJul],
    format=Documenter.HTML(prettyurls=get(ENV,"CI","false")=="true",
                          canonical="https://triscale-innov.github.io/LidJul.jl/",
                          edit_link="master"),
    checkdocs=:exports,
    doctest=true,
    sitename="LidJul.jl",
    pages=["Home"=>"index.md", "Solvers"=>"solvers.md", "Neumann problems"=>"neumann.md",
           "Cavity simulation"=>"cavity.md", "Validation"=>"validation.md",
           "Benchmarks"=>"benchmarks.md", "Migration"=>"migration.md", "Modernization record"=>"modernization.md", "API"=>"api.md"],
)
