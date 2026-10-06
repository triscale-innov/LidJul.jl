"""
    PoissonSparseIterative(laplacian)
    PoissonSparseIterative(matrix; neumann=...)

Sparse CG or stationary iteration solver. Select `method="cg"`, `"jacobi"`,
`"gauss_seidel"`, `"sor"`, or `"ssor"` in `solve!`. All methods use the same
physical stopping test and return `SolveResult`. Stationary iterations perform
one sweep per iteration (forward/backward for SSOR) and reuse vector workspaces.
Jacobi is damped by 2/3 for Neumann problems to suppress the alternating mode. Pure Neumann solutions are
recentered after each sweep.
"""
struct PoissonSparseIterative{T<:AbstractFloat,P} <: AbstractPoissonSolver{T}
    operator::SparseMatrixCSC{T,Int}
    preconditioner::P
    shape::Tuple{Int,Int}
    allneumann::Bool
    residual::Vector{T}
    workx::Vector{T}
    workb::Vector{T}
end
function PoissonSparseIterative(a::SparseMatrixCSC{T};neumann=_has_constant_nullspace(a),shape=(size(a,1),1)) where T
    original,_=_matrix_and_anchor(a,neumann)
    prod(shape)==size(a,1) || throw(DimensionMismatch("invalid operator shape"))
    PoissonSparseIterative(original,IterativeSolvers.Identity(),shape,neumann,
                           zeros(T,size(a,1)),zeros(T,size(a,1)),zeros(T,size(a,1)))
end
PoissonSparseIterative(a::Laplacian2D)=PoissonSparseIterative(sparse(a);neumann=a.allneumann,shape=size(a))
function solve!(x,b,s::PoissonSparseIterative{T};method="cg",reltol=_default_reltol(T),abstol=zero(T),maxiter=1000,store_history=true) where T
    method in ("cg","jacobi","gauss_seidel","sor","ssor") || throw(ArgumentError("unknown iterative method: $method"))
    method=="cg" && return _krylov_solve!(x,b,s;reltol,abstol,maxiter,store_history)
    threshold=_prepare(x,b,s;reltol,abstol,maxiter)
    initial,history=_initial(x,b,s,threshold,store_history,maxiter)
    initial===nothing || return initial
    copyto!(s.workx,vec(x)); copyto!(s.workb,vec(b))
    omega=T(2)/(one(T)+sin(T(π)/sqrt(T(length(x)))))
    iterations=0
    for k in 1:maxiter
        _stationary_sweep!(s,method,omega)
        s.allneumann && _recenter!(s.workx)
        copyto!(vec(x),s.workx)
        r=residual_norm(x,b,s)
        store_history && push!(history,r)
        iterations=k
        r<=threshold && break
    end
    _result(x,b,s,iterations,history,threshold)
end
solve!(x,b,s::PoissonSparseIterative,method::AbstractString;kwargs...)=solve!(x,b,s;method,kwargs...)

function _stationary_sweep!(s,method,omega)
    x,b,a,r=s.workx,s.workb,s.operator,s.residual
    mul!(r,a,x)
    @. r=b-r
    if method=="jacobi"
        # Damping removes the alternating nondecaying mode of a Neumann grid.
        weight=s.allneumann ? eltype(x)(2/3) : one(eltype(x))
        for i in eachindex(x)
            x[i]+=weight*r[i]/a[i,i]
        end
        return
    end
    weight=method=="gauss_seidel" ? one(eltype(x)) : omega
    _coordinate_sweep!(x,r,a,weight,1:length(x))
    method=="ssor" && _coordinate_sweep!(x,r,a,weight,length(x):-1:1)
end
function _coordinate_sweep!(x,r,a,weight,indices)
    for col in indices
        delta=weight*r[col]/a[col,col]
        x[col]+=delta
        @inbounds for k in nzrange(a,col)
            r[a.rowval[k]]-=a.nzval[k]*delta
        end
    end
end
