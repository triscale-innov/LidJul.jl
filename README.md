# LidJul.jl

Poisson solvers and a headless lid-driven cavity simulation for Julia 1.13.
The core supports Float32/Float64, rectangular grids, mixed boundary conditions
and consistent zero-mean Neumann solutions.

```julia
using LidJul
lap = Laplacian2D(32, 64, 1, 2, dirichlet, neumann, dirichlet, neumann)
x = zeros(32, 64)
result = solve!(x, ones(32, 64), PoissonGMG(lap); reltol=1e-8)
cavity = simulate_cavity(CavityConfig(Re=100, nx=32, dt=0.005, tf=20))
```

Install and test from this checkout:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --startup-file=no --project=. test/cli_smoke.jl
```

The English documentation uses **Documenter.jl**, including executable examples,
API docstrings, migration guidance, numerical validation and benchmark instructions:

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Open `docs/build/index.html`. Source documentation starts at
[docs/src/index.md](docs/src/index.md).
GitHub Actions runs the command-line examples on Linux, macOS and Windows,
builds the documentation with doctests, and uploads the generated HTML as the
`documentation` artifact.

Optional plotting uses the `examples/` environment and Julia package extensions:
load `CairoMakie` for `plot_cavity`, or `PlotlyBase` for `plot_interactive`.
`examples/interactive/` isolates the optional GLMakie desktop backend. The
`benchmark/` environment measures setup/solve times, allocations and convergence.
The published-reference study is in `validation/`; historical experiments are
preserved, unsupported, in `archive/`.

Version 0.3 changes constructors to return solvers directly and standardizes
`solve!` on `SolveResult`. See [migration](docs/src/migration.md).
