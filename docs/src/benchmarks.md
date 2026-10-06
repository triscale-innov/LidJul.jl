```@meta
CurrentModule = LidJul
DocTestSetup = :(using LidJul)
```

# Benchmarks

`benchmark/benchmarks.jl` exposes `benchmark_suite` and `benchmark_report`.
Construction and repeated solves are measured separately. Each solve sample
resets the initial iterate to zero with `evals=1`, uses cached solver storage, and
excludes history allocation. Every case first passes a physical-residual check.

```sh
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
```

```julia
using LinearAlgebra
BLAS.set_num_threads(1)
include("benchmark/benchmarks.jl")
report = benchmark_report(seconds=0.2, samples=30)
```

Defaults cover 16×32, 32×64 and 64×64 grids, Dirichlet/mixed/Neumann boundaries,
six solvers, both scalar types and two tolerances per type. Adapt `grids`, `types`,
`tolerances`, `maxiter`, `seconds` and `samples` to the question being studied.
Float32 tolerances are deliberately larger than Float64 tolerances.

The TOML report records setup/solve times, bytes and allocation counts, samples,
iterations, achieved residuals, grid dimensions, boundary conditions, requested
tolerances, resolved and actually loaded Julia/package versions, CPU, operating system and BLAS configuration.
The statistic is the minimum warmed trial. Use larger sample budgets and a quiet
machine before drawing performance conclusions; a smoke run validates the
measurement machinery but does not establish a ranking.

The old README timing figures are historical and have been removed from the
current performance claims. Tensor transforms and multigrid have different setup
and asymptotic costs; results should be compared at an equal physical tolerance
and with construction amortization stated explicitly.

## Executed local run

The checked-in `benchmark/results.toml` contains 216 accuracy-checked cases with
up to 20 warmed samples per construction/solve trial, a 0.1-second trial budget and one
BLAS thread. Both Float32 and Float64, three grids, three boundary sets, six
methods and two tolerances per type were exercised. This validates the complete
suite and allocation reporting. The sampled minima are local measurements and
should not be generalized to other machines or used as a universal ranking.

The report records loaded versions separately from resolved versions. A REPL
integration can preload a different version of a utility package; the distinction
makes that circumstance explicit rather than implying that the manifest alone
fully describes the running process.
