```@meta
CurrentModule = LidJul
DocTestSetup = :(using LidJul)
```

# Solvers

All Poisson solvers use `solve!(x, b, solver; reltol, abstol, maxiter,
store_history)` and return the same result type. Convergence means
`norm(A*x-b) <= max(abstol, reltol*norm(b))`, measured with the original operator.
A failed convergence flag is meaningful even for a direct solver when a requested
tolerance is below the attainable floating-point accuracy.

| Solver | Algorithm | Typical use |
|:--|:--|:--|
| `PoissonTTSolver` | Separable eigenvector transforms | Constant coefficients on rectangular grids |
| `TensorialOperator` | Separable shifted operator | Implicit diffusion / Helmholtz equations |
| `PoissonSparseLU` | Cached sparse factorization | Reference and repeated sparse solves |
| `PoissonSparseAMG` | CG with cached AMG preconditioner | Larger symmetric Poisson systems |
| `PoissonSparseCGILU` | GMRES with cached incomplete LU | Sparse systems with an ILU preconditioner |
| `PoissonSparseIterative` | CG or stationary sweeps | Comparison and teaching |
| `PoissonGMG` | Geometric multigrid V-cycles | Power-of-two rectangular grids |

The historical `PoissonSparseCGILU` name is retained, but its implementation uses
GMRES: an incomplete LU factorization need not be symmetric, as required by CG.
AMG supports `algorithm=:ruge_stuben` and `:smoothed_aggregation`.

Iterative solvers use the incoming `x` as the initial iterate. `maxiter=0` only
checks it. For direct methods one iteration denotes one factorized solve; for
multigrid it denotes one V-cycle. `history` includes the initial residual when
requested; its last value is always the physical final residual. CG history
contains recursively updated residuals, which can differ slightly from freshly
computed residuals because of roundoff.

## Types, dimensions and reuse

```julia
lap = Laplacian2D(32, 64, 1, 2, dirichlet, neumann, dirichlet, neumann; T=Float32)
solver = PoissonGMG(lap)
x = zeros(Float32, 32, 64)
result = solve!(x, ones(Float32, 32, 64), solver; reltol=1f-4, store_history=false)
```

Tensor and multigrid methods require matrices with the spatial grid dimensions.
Sparse methods accept compatible vectors or matrices, in Julia's column-major
ordering. Solution and RHS must use the solver's scalar type and must not alias.
Sparse LU uses SuiteSparse's `Float64` factorization workspace for `Float32`
inputs, then converts the solution back. Other methods retain the chosen scalar
type in their numerical workspaces.

Geometric multigrid supports independent grid spacings, all 16 combinations of
homogeneous boundary conditions and `Float32`/`Float64`. Both dimensions must
be powers of two and at least four. Each level has one solution, RHS and residual
buffer. The coarse problem uses the tensor direct solver.

Solvers own mutable scratch storage. Use a distinct solver in each concurrent
task; sharing one instance between simultaneous calls is unsupported. Reusing
separate solvers in concurrent tasks is covered by the tests.
