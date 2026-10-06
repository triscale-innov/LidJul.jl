```@meta
CurrentModule = LidJul
DocTestSetup = :(using LidJul)
```

# LidJul.jl

LidJul solves cell-centered Poisson problems and the two-dimensional
lid-driven cavity on a staggered MAC grid. The numerical core runs without a
graphics backend and supports Julia 1.13, `Float32`, and `Float64`.

## Installation

From a checkout of this repository:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Use Julia's current stable release through `juliaup update release`. This
migration was verified with Julia 1.13.1. Dependencies are specified by compatible
version ranges; the manifests record the exact versions used locally.

## First solve

```jldoctest
julia> lap = Laplacian2D(8, 16, 1, 2, dirichlet, dirichlet, dirichlet, dirichlet);

julia> x = zeros(8, 16); b = ones(8, 16);

julia> result = solve!(x, b, PoissonTTSolver(lap); reltol=1e-10);

julia> result.converged && result.iterations == 1
true
```

The constructor returns a solver, and `solve!` returns a [`SolveResult`](@ref).
Setup work is cached. Reuse the same solver when the operator stays unchanged.

## Environments

| Environment | Purpose |
|:--|:--|
| Repository root | Numerical package and its tests |
| `docs/` | Documenter documentation and doctests |
| `examples/` | Optional PlotlyBase and CairoMakie visualization |
| `examples/interactive/` | Optional desktop OpenGL backend |
| `benchmark/` | Reproducible timing and allocation measurements |
| `archive/` | Historical experiments preserved for provenance |

Build this documentation with:

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The build checks all exported docstrings and treats documentation errors as
failures. Open `docs/build/index.html` after a successful build.
