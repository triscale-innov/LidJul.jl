# Multigrid design inspired by Harald Köstler, Multigrid HowTo (2008).
"""
    maxlevels(nx[, ny])

Maximum number of geometric multigrid levels, including a two-cell coarse
axis. Both dimensions must be powers of two and at least four.
"""
function maxlevels(nx::Integer,ny::Integer=nx)
    all(n->n>=4 && ispow2(n),(nx,ny)) ||
        throw(ArgumentError("grid dimensions must be powers of two, at least four"))
    min(trailing_zeros(nx),trailing_zeros(ny))
end

"""
    PoissonGMG(laplacian[, GSSmoother]; nlevels=maxlevels(nx,ny), nprae=2, npost=2)

Cached geometric multigrid V-cycles with red-black smoothing, cell-average
restriction, piecewise constant prolongation and an exact coarse solve.
Supports rectangular domains, mixed boundaries, and `Float32`/`Float64`.
Both grid dimensions must be powers of two and at least four. Separate solver
instances are required for concurrent calls. `iterations` counts V-cycles.
"""
struct PoissonGMG{T<:AbstractFloat,C} <: AbstractPoissonSolver{T}
    levels::Vector{Laplacian2D{T}}
    smoothers::Vector{GSSmoother{T}}
    solutions::Vector{Matrix{T}}
    right_hand_sides::Vector{Matrix{T}}
    residuals::Vector{Matrix{T}}
    coarse::C
    shape::Tuple{Int,Int}
    allneumann::Bool
    residual::Matrix{T}
    nprae::Int
    npost::Int
end
function PoissonGMG(a::Laplacian2D{T},::Type{GSSmoother}=GSSmoother;
                    nlevels=maxlevels(a.nx,a.ny),nprae=2,npost=2) where T
    limit=maxlevels(a.nx,a.ny)
    1<=nlevels<=limit || throw(ArgumentError("invalid number of multigrid levels"))
    nprae isa Integer && npost isa Integer && min(nprae,npost)>=0 ||
        throw(ArgumentError("smoothing counts must be nonnegative integers"))
    nprae+npost>0 || throw(ArgumentError("at least one smoothing sweep is required"))
    levels=[Laplacian2D(a.nx÷2^(k-1),a.ny÷2^(k-1),a.Lx,a.Ly,a.bc[1]...,a.bc[2]...;T) for k=1:nlevels]
    solutions=[zeros(T,size(l)) for l in levels]
    rhs=[zeros(T,size(l)) for l in levels]
    residuals=[zeros(T,size(l)) for l in levels[1:end-1]]
    coarse=PoissonTTSolver(levels[end])
    PoissonGMG(levels,GSSmoother.(levels),solutions,rhs,residuals,coarse,size(a),
               a.allneumann,isempty(residuals) ? zeros(T,size(a)) : first(residuals),nprae,npost)
end
function _stencil!(out,a::Laplacian2D,x)
    nx,ny=size(a)
    cx,cy=a.lpx.dxm2_,a.lpy.dxm2_
    @inbounds for j=1:ny,i=1:nx
        value=(a.lpx[i,i]+a.lpy[j,j])*x[i,j]
        i>1 && (value-=cx*x[i-1,j])
        i<nx && (value-=cx*x[i+1,j])
        j>1 && (value-=cy*x[i,j-1])
        j<ny && (value-=cy*x[i,j+1])
        out[i,j]=value
    end
    out
end
_apply!(out,s::PoissonGMG,x)=_stencil!(out,s.levels[1],x)
function _vcycle!(s,level)
    x,b=s.solutions[level],s.right_hand_sides[level]
    if level==length(s.levels)
        fill!(x,0)
        solve!(x,b,s.coarse;reltol=eps(eltype(x)),store_history=false)
        return
    end
    for _=1:s.nprae
        _smooth!(x,b,s.smoothers[level])
    end
    r=s.residuals[level]
    _stencil!(r,s.levels[level],x)
    r .= b .- r
    bc=s.right_hand_sides[level+1]
    @inbounds for j=1:size(bc,2),i=1:size(bc,1)
        fi,fj=2i-1,2j-1
        bc[i,j]=(r[fi,fj]+r[fi+1,fj]+r[fi,fj+1]+r[fi+1,fj+1])/4
    end
    s.allneumann && _recenter!(bc)
    xc=s.solutions[level+1]
    fill!(xc,0)
    _vcycle!(s,level+1)
    @inbounds for j=1:size(xc,2),i=1:size(xc,1)
        fi,fj=2i-1,2j-1
        correction=xc[i,j]
        x[fi,fj]+=correction
        x[fi+1,fj]+=correction
        x[fi,fj+1]+=correction
        x[fi+1,fj+1]+=correction
    end
    for _=1:s.npost
        _smooth!(x,b,s.smoothers[level])
    end
end
function solve!(x,b,s::PoissonGMG{T};reltol=_default_reltol(T),abstol=zero(T),maxiter=1000,store_history=true) where T
    size(x)==size(b)==s.shape || throw(DimensionMismatch("multigrid requires matrices with size $(s.shape)"))
    threshold=_prepare(x,b,s;reltol,abstol,maxiter)
    initial,history=_initial(x,b,s,threshold,store_history,maxiter)
    initial===nothing || return initial
    copyto!(s.solutions[1],x)
    copyto!(s.right_hand_sides[1],b)
    iterations=0
    for k=1:maxiter
        _vcycle!(s,1)
        copyto!(x,s.solutions[1])
        s.allneumann && _recenter!(x)
        r=residual_norm(x,b,s)
        store_history && push!(history,r)
        iterations=k
        r<=threshold && break
    end
    _result(x,b,s,iterations,history,threshold)
end

"""
    PoissonGMG_new(args...; kwargs...)

Compatibility constructor returning [`PoissonGMG`](@ref). The former duplicate
implementation has been consolidated into the same tested solver.
"""
PoissonGMG_new(args...;kwargs...)=PoissonGMG(args...;kwargs...)
