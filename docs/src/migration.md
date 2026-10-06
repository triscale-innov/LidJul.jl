```@meta
CurrentModule = LidJul
DocTestSetup = :(using LidJul)
```

# Migration to 0.3

The numerical package now targets Julia 1.13 and the current compatible package
releases. Plotting, example utilities and benchmarking have separate environments.
PlotlyBase and Makie integrations are optional Julia package extensions.

| Previous API | Current API |
|:--|:--|
| `solver, timing = PoissonSparseLU(...)` | `solver = PoissonSparseLU(lap)`; measure setup separately |
| Method-specific `solve!` return values | `result = solve!(x,b,solver)` with `SolveResult` |
| Implicit fixed stopping thresholds | `reltol`, `abstol`, `maxiter`, `store_history` |
| `PoissonGMG_new` duplicate implementation | Compatibility constructor for the unified `PoissonGMG` |
| Neumann diagonal correction as input | Pass `lap` directly and use its zero-mean gauge |
| `test/mit_implicit.jl` simulation script | Parameterized `CavityConfig` and `simulate_cavity` |
| Plotting imported with numerical code | Load `PlotlyBase` or a Makie backend explicitly |

Historical experimental implementations are preserved in `archive/` and excluded
from normal imports, tests and documentation builds. They are unsupported and may
use obsolete APIs. The active public API and examples have English documentation
and comments. Author attribution is retained in archived sources.

The former unrestricted `Base.size` methods, rectangular tensor-transform bug,
boundary-side bookkeeping and obsolete Krylov tolerance keywords have been fixed.
The new tests cover mixed boundaries, rectangular operators, scalar types,
Neumann compatibility, scientific refinement and solver reuse.
