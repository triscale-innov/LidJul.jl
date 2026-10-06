```@meta
CurrentModule = LidJul
```

# Modernization record

The initial project was a numerical research workspace built with Julia 1.11.6.
Its Poisson solvers used inconsistent return values and stopping criteria, and
its test entry point ran a long graphical cavity experiment without numerical
assertions. The supported package now targets Julia 1.13, verified locally on
Julia 1.13.1, with headless scientific tests and independent examples.

## Completed recommendations

| Audit recommendation | Implemented change |
|:--|:--|
| Common solver contract | `SolveResult`, consistent tolerances, iteration limits and physical residuals |
| Cavity validation | Parameterized MAC solver, incremental projection, wall/divergence tests and offline Ghia comparison with grid/timestep studies |
| Separate numerical and graphical code | Dedicated environments and optional Plotly/Makie package extensions |
| Typing and memory | Typed factors/hierarchies, cached AMG preconditioner, reusable LU conversion/output buffers and one rectangular Float32/Float64 GMG |
| Neumann policy | RHS compatibility checks at every scale, zero-mean gauge and original-operator residuals |
| Reproducible performance comparisons | Separate construction/solve trials, allocations, equal tolerances, accuracy checks and complete version/BLAS metadata |
| API and experiment cleanup | Historical sources archived, modern examples and English Documenter API coverage/doctests |

The migration also fixed swapped boundary metadata, unrestricted extensions of
`Base.size`, rectangular sparse-neighbor indexing, tensor-transform ordering and
obsolete Krylov keywords. The multigrid smoother now incorporates boundary
coefficients directly in its diagonal and treats each grid spacing separately.
Stationary sweeps use cached residual updates; Neumann Jacobi uses damping to
remove its alternating mode. ILU uses GMRES with right preconditioning.

## Package versions used

| Package | Initial version | Updated version | Environment |
|:--|:--|:--|:--|
| AlgebraicMultigrid | 0.5.1 | 2.0.2 | Core |
| IncompleteLU | 0.2.1 | 0.2.1 | Core |
| IterativeSolvers | 0.9.4 | 0.9.4 | Core |
| Documenter | 0.22.4 | 1.19.0 | Documentation |
| BenchmarkTools | 1.6.0 | 1.8.0 | Benchmarks |
| GLMakie | 0.13.6 | 0.13.15 | Interactive examples |
| CairoMakie | — | 0.15.15 | Examples |
| Makie | 0.24.6 | 0.24.15 | Optional graphics |
| PlotlyBase | 0.8.21 | 0.10.0 | Optional graphics |
| Plots | 1.40.19 | 1.41.7 | Examples |
| JLD2 | 0.6.1 | 0.6.7 | Examples |
| PrettyTables | 3.0.8 | 3.5.0 | Examples |
| DataStructures | 0.19.1 | 0.19.6 | Examples |
| SIMD | 3.7.1 | 3.7.2 | Examples |
| StaticArrays | 1.9.15 | 1.9.22 | Examples |
| Interpolations | 0.16.2 | 0.16.3 | Examples |

LoopVectorization, Preconditioners and PlotlyJS were removed. Their old constraints
prevented current AMG/PlotlyBase versions, and the maintained code uses native
loops, AlgebraicMultigrid's preconditioner API and PlotlyBase directly.
Compatibility ranges permit subsequent compatible updates; manifests pin the
actual resolved environments. Upstream compatibility constraints can hold
transitive packages below their latest release.

Julia's standard libraries LinearAlgebra and SparseArrays are version 1.13.0 in
the Julia 1.13.1 installation. Their version numbers need not match Julia's patch
number. See [Migration to 0.3](@ref) before updating callers.
