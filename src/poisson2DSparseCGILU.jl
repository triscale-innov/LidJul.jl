"""
    PoissonSparseCGILU(laplacian; droptol=0.2)
    PoissonSparseCGILU(matrix; droptol=0.2, neumann=...)

Krylov solver with a typed, cached incomplete LU factorization. The historical
name is retained, but GMRES is used because ILU is generally nonsymmetric and
does not satisfy CG's symmetric-positive-definite preconditioner requirement.
Pure Neumann problems retain the physical operator and use a zero-mean gauge.
"""
struct PoissonSparseCGILU{T<:AbstractFloat,P} <: AbstractPoissonSolver{T}
    operator::SparseMatrixCSC{T,Int}
    preconditioner::P
    shape::Tuple{Int,Int}
    allneumann::Bool
    residual::Vector{T}
end
function PoissonSparseCGILU(a::SparseMatrixCSC{T};droptol=0.2,
                            neumann=_has_constant_nullspace(a),shape=(size(a,1),1)) where T
    isfinite(droptol) && droptol>=0 || throw(ArgumentError("droptol must be finite and nonnegative"))
    original,anchored=_matrix_and_anchor(a,neumann)
    prod(shape)==size(a,1) || throw(DimensionMismatch("invalid operator shape"))
    PoissonSparseCGILU(original,IncompleteLU.ilu(anchored;τ=T(droptol)),shape,neumann,zeros(T,size(a,1)))
end
PoissonSparseCGILU(a::Laplacian2D;kwargs...)=PoissonSparseCGILU(sparse(a);neumann=a.allneumann,shape=size(a),kwargs...)
function solve!(x,b,s::PoissonSparseCGILU{T};reltol=_default_reltol(T),abstol=zero(T),maxiter=1000,store_history=true) where T
    _krylov_solve!(x,b,s;reltol,abstol,maxiter,store_history,method=:gmres)
end
