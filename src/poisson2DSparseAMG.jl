"""
    PoissonSparseAMG(laplacian; algorithm=:ruge_stuben)
    PoissonSparseAMG(matrix; algorithm=:ruge_stuben, neumann=...)

Conjugate gradients with a cached algebraic multigrid hierarchy and preconditioner.
`algorithm` is `:ruge_stuben` or `:smoothed_aggregation`. Setup is performed once.
For pure Neumann problems the preconditioner is anchored, while the physical
operator is kept singular and the solution uses the common zero-mean gauge.
"""
struct PoissonSparseAMG{T<:AbstractFloat,H,P} <: AbstractPoissonSolver{T}
    operator::SparseMatrixCSC{T,Int}
    hierarchy::H
    preconditioner::P
    shape::Tuple{Int,Int}
    allneumann::Bool
    residual::Vector{T}
end
function PoissonSparseAMG(a::SparseMatrixCSC{T};algorithm=:ruge_stuben,
                          neumann=_has_constant_nullspace(a),shape=(size(a,1),1)) where T
    original,anchored=_matrix_and_anchor(a,neumann)
    prod(shape)==size(a,1) || throw(DimensionMismatch("invalid operator shape"))
    hierarchy=algorithm===:ruge_stuben ? AlgebraicMultigrid.ruge_stuben(anchored) :
        algorithm===:smoothed_aggregation ? AlgebraicMultigrid.smoothed_aggregation(anchored) :
        throw(ArgumentError("unknown AMG algorithm: $algorithm"))
    PoissonSparseAMG(original,hierarchy,AlgebraicMultigrid.aspreconditioner(hierarchy),shape,
                     neumann,zeros(T,size(a,1)))
end
PoissonSparseAMG(a::Laplacian2D;kwargs...)=PoissonSparseAMG(sparse(a);neumann=a.allneumann,shape=size(a),kwargs...)
function _krylov_solve!(x,b,s;reltol,abstol,maxiter,store_history,method=:cg)
    threshold=_prepare(x,b,s;reltol,abstol,maxiter)
    initial,history=_initial(x,b,s,threshold,store_history,maxiter)
    initial===nothing || return initial
    algorithm=method===:cg ? IterativeSolvers.cg! : IterativeSolvers.gmres!
    _,log = if method===:cg
        algorithm(vec(x),s.operator,vec(b);Pl=s.preconditioner,reltol=0,abstol=threshold,maxiter,log=true)
    else
        algorithm(vec(x),s.operator,vec(b);Pr=s.preconditioner,reltol=0,abstol=threshold,maxiter,log=true)
    end
    store_history && append!(history,log.data[:resnorm])
    _result(x,b,s,log.iters,history,threshold)
end
function solve!(x,b,s::PoissonSparseAMG{T};reltol=_default_reltol(T),abstol=zero(T),maxiter=1000,store_history=true) where T
    _krylov_solve!(x,b,s;reltol,abstol,maxiter,store_history)
end
