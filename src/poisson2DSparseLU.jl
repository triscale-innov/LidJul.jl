"""
    PoissonSparseLU(laplacian)
    PoissonSparseLU(matrix; neumann=...)

Sparse direct solver with a typed, cached LU factorization. Matrix construction
infers a constant null space unless `neumann` is supplied. For pure Neumann
problems, only the factored matrix is anchored: residuals use the original
operator and solutions are recentered. Float32 uses UMFPACK's Float64
factorization with a reusable conversion buffer. The constructor returns a solver,
not the historical `(solver, elapsed_time)` tuple.
"""
struct PoissonSparseLU{T<:AbstractFloat,F,V<:AbstractVector} <: AbstractPoissonSolver{T}
    operator::SparseMatrixCSC{T,Int}
    factorization::F
    work::V
    rhs_work::V
    shape::Tuple{Int,Int}
    allneumann::Bool
    residual::Vector{T}
end
function _matrix_and_anchor(a::SparseMatrixCSC{T},neumann) where T
    size(a,1)==size(a,2) || throw(DimensionMismatch("operator must be square"))
    T in (Float32,Float64) || throw(ArgumentError("supported types are Float32 and Float64"))
    all(isfinite,nonzeros(a)) || throw(ArgumentError("operator must be finite"))
    original=copy(a)
    anchored=copy(a)
    neumann && (anchored[1,1]+=max(abs(anchored[1,1]),one(T)))
    original,anchored
end
function PoissonSparseLU(a::SparseMatrixCSC{T};neumann=_has_constant_nullspace(a),shape=(size(a,1),1)) where T
    original,anchored=_matrix_and_anchor(a,neumann)
    prod(shape)==size(a,1) || throw(DimensionMismatch("invalid operator shape"))
    factor=lu(anchored)
    PoissonSparseLU(original,factor,zeros(eltype(factor),size(a,1)),zeros(eltype(factor),size(a,1)),shape,neumann,zeros(T,size(a,1)))
end
PoissonSparseLU(a::Laplacian2D)=PoissonSparseLU(sparse(a);neumann=a.allneumann,shape=size(a))
function solve!(x,b,s::PoissonSparseLU{T};reltol=_default_reltol(T),abstol=zero(T),maxiter=1000,store_history=true) where T
    threshold=_prepare(x,b,s;reltol,abstol,maxiter)
    initial,history=_initial(x,b,s,threshold,store_history,maxiter)
    initial===nothing || return initial
    copyto!(s.rhs_work,vec(b))
    ldiv!(s.work,s.factorization,s.rhs_work)
    copyto!(vec(x),s.work)
    store_history && push!(history,zero(T))
    _result(x,b,s,1,history,threshold)
end
