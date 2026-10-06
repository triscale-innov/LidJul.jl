"""
    PoissonTTSolver(laplacian)

Build a direct tensor solver for a [`Laplacian2D`](@ref). Construction diagonalizes
the two one-dimensional operators; subsequent solves reuse all workspaces.
Rectangular grids, Float32/Float64 and pure Neumann conditions are supported.
"""
struct PoissonTTSolver{T<:AbstractFloat,S<:TensorialOperator{T}} <: AbstractPoissonSolver{T}
    tensor::S
end
PoissonTTSolver(a::Laplacian2D)=PoissonTTSolver(TensorialOperator(SymTridiagonal(a.lpx),SymTridiagonal(a.lpy)))
solve!(x,b,s::PoissonTTSolver;kwargs...)=solve!(x,b,s.tensor;kwargs...)
residual_norm(x,b,s::PoissonTTSolver)=residual_norm(x,b,s.tensor)
_solver_size(s::PoissonTTSolver)=s.tensor.shape
_is_neumann(s::PoissonTTSolver)=s.tensor.allneumann
