```@meta
CurrentModule = LidJul
DocTestSetup = :(using LidJul)
```

# Neumann problems

A Laplacian with homogeneous Neumann conditions on all four sides is singular:
adding a constant to the solution does not change its image. A solution exists
only when the discrete RHS has zero sum.

Every solver validates this compatibility condition with a roundoff-scaled
threshold and returns a solution whose mean is zero. It rejects an incompatible
RHS instead of silently discarding a physical source term.

```jldoctest
julia> lap = Laplacian2D(4, 8, 1, 2, neumann, neumann, neumann, neumann);

julia> b = zeros(4, 8); b[1,1] = 1; b[end,end] = -1;

julia> x = zeros(4, 8); r = solve!(x, b, PoissonSparseLU(lap); reltol=1e-10);

julia> r.converged && abs(sum(x)/length(x)) < 1e-12
true
```

Sparse LU anchors one diagonal internally. AMG and ILU anchor only their
preconditioners. Krylov iterations and all final residual measurements use the
original singular operator. Tensor methods remove the constant eigenmode, while
multigrid recenters its corrections. These choices produce a consistent gauge
and allow meaningful cross-solver comparisons.

Use `sparse(lap)` when checking `norm(A*x-b)`. The legacy `sparse_corr(lap)`
helper returns an anchored operator and therefore changes the physical problem.
Prefer constructing a solver directly from `lap`.
