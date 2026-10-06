"""Abstract superclass of mutable-workspace solvers with scalar type `T`."""
abstract type AbstractPoissonSolver{T<:AbstractFloat} end

"""
    SolveResult(converged, iterations, residual_norm, history)

Common return value of [`solve!`](@ref). `residual_norm` measures `norm(A*x-b)`
with the original operator. `history` contains the initial residual and recorded
iteration residuals when `store_history=true`, otherwise it is empty. Direct
solvers perform at most one iteration. The solution is written into `x`.
"""
struct SolveResult{T<:AbstractFloat}
    converged::Bool
    iterations::Int
    residual_norm::T
    history::Vector{T}
end

"""
    solve!(x, b, solver; reltol=sqrt(eps(T)), abstol=0, maxiter=1000,
           store_history=true, method="cg")

Overwrite `x` with an approximate solution and return [`SolveResult`](@ref).
Stop when the physical residual is at most `max(abstol, reltol*norm(b))`.
`x` is the initial iterate for iterative solvers. The arrays must have the solver's
scalar type and compatible dimensions. Pure Neumann right-hand sides must have
zero sum; all solvers return a zero-mean solution. `maxiter=0` only checks the
initial iterate. Workspaces are reused: use a separate solver per concurrent task.
`method` is supported only by [`PoissonSparseIterative`](@ref).
"""
function solve! end

_default_reltol(::Type{T}) where T = sqrt(eps(T))
_scalar_type(::AbstractPoissonSolver{T}) where T = T
_is_neumann(s) = s.allneumann
_solver_size(s) = s.shape
_recenter!(x) = (x .-= sum(x)/length(x); x)
function _has_constant_nullspace(a::AbstractMatrix{T}) where T
    norm(a*ones(T,size(a,2)),Inf) <= 100eps(T)*opnorm(a,Inf)
end
function _prepare(x,b,s;reltol,abstol,maxiter)
    T=_scalar_type(s)
    size(x)==size(b) || throw(DimensionMismatch("solution and RHS sizes differ"))
    length(x)==prod(_solver_size(s)) || throw(DimensionMismatch("solver and array sizes differ"))
    eltype(x)==eltype(b)==T || throw(ArgumentError("solution and RHS must have scalar type $T"))
    Base.mightalias(x,b) && throw(ArgumentError("solution and RHS must not alias"))
    all(isfinite,x) && all(isfinite,b) || throw(ArgumentError("solution and RHS must be finite"))
    isfinite(reltol) && reltol>=0 && isfinite(abstol) && abstol>=0 ||
        throw(ArgumentError("tolerances must be finite and nonnegative"))
    maxiter isa Integer && maxiter>=0 || throw(ArgumentError("maxiter must be a nonnegative integer"))
    if _is_neumann(s)
        abs(sum(b)) <= 100eps(T)*sum(abs,b) ||
            throw(ArgumentError("pure Neumann RHS must have zero sum"))
        _recenter!(x)
    end
    T(max(abstol,reltol*norm(b)))
end

"""
    residual_norm(x, b, solver)

Compute the physical residual norm using the solver's cached workspace. For
Neumann problems this always uses the original singular operator.
"""
function residual_norm(x,b,s::AbstractPoissonSolver)
    size(x)==size(b) && length(x)==prod(_solver_size(s)) || throw(DimensionMismatch("invalid residual dimensions"))
    _apply!(s.residual,s,x)
    for i in eachindex(s.residual)
        s.residual[i]-=b[i]
    end
    norm(s.residual)
end
function _result(x,b,s,iterations,history,threshold)
    _is_neumann(s) && _recenter!(x)
    final=residual_norm(x,b,s)
    isempty(history) || (history[end]=final)
    SolveResult(final<=threshold,iterations,final,history)
end
function _initial(x,b,s,threshold,store_history,maxiter)
    r=residual_norm(x,b,s)
    history=store_history ? [r] : typeof(r)[]
    if r<=threshold || maxiter==0
        return SolveResult(r<=threshold,0,r,history),history
    end
    nothing,history
end
function _apply!(out,s,x)
    mul!(vec(out),s.operator,vec(x))
    out
end
