"""
    TensorialOperator(Ax::SymTridiagonal, Ay::SymTridiagonal; shift=0)
    TensorialOperator(Array, T, nx, ny, hx, hy, clx, cly, αx, αy, shift)

Direct separable solver for `Ax*X + X*Ay + shift*X = B`, with cached transforms
and workspaces. The legacy constructor uses endpoint diagonal codes 1 (Neumann),
2 (node-centered Dirichlet), and 3 (cell-centered Dirichlet). Supports Float32 and
Float64. Symmetric eigenvector transforms use transposes, not explicit inverses.
A singular Neumann operator selects a zero-mean solution.
"""
struct TensorialOperator{T<:AbstractFloat} <: AbstractPoissonSolver{T}
    ax::SymTridiagonal{T,Vector{T}}
    ay::SymTridiagonal{T,Vector{T}}
    mx::Matrix{T}
    my::Matrix{T}
    inverse_spectrum::Matrix{T}
    shift::T
    shape::Tuple{Int,Int}
    allneumann::Bool
    t1::Matrix{T}
    t2::Matrix{T}
    residual::Matrix{T}
end
function TensorialOperator(ax::SymTridiagonal{T},ay::SymTridiagonal{T};shift=zero(T)) where {T<:AbstractFloat}
    T in (Float32,Float64) || throw(ArgumentError("supported types are Float32 and Float64"))
    isfinite(shift) && shift>=0 || throw(ArgumentError("shift must be finite and nonnegative"))
    all(isfinite,ax) && all(isfinite,ay) || throw(ArgumentError("operators must be finite"))
    ex,ey=eigen(ax),eigen(ay)
    nx,ny=size(ax,1),size(ay,1)
    alln=shift==0 && _has_constant_nullspace(ax) && _has_constant_nullspace(ay)
    invd=Matrix{T}(undef,nx,ny)
    for j in 1:ny,i in 1:nx
        d=ex.values[i]+ey.values[j]+T(shift)
        if alln && i==j==1
            invd[i,j]=zero(T)
        else
            d>0 || throw(ArgumentError("the separable operator must be positive semidefinite"))
            invd[i,j]=inv(d)
        end
    end
    TensorialOperator{T}(ax,ay,ex.vectors,ey.vectors,invd,T(shift),(nx,ny),alln,
                        zeros(T,nx,ny),zeros(T,nx,ny),zeros(T,nx,ny))
end
function K2(n::Integer,h::Real,first,last;T=Float64)
    n>=2 && isfinite(h) && h>0 || throw(ArgumentError("n must be at least 2 and h positive"))
    d=fill(T(2),n); d[1],d[end]=T(first),T(last)
    SymTridiagonal(d./T(h)^2,fill(-inv(T(h)^2),n-1))
end
K1(n,h,code;T=Float64)=K2(n,h,code,code;T)
function TensorialOperator(arraytype,::Type{T},nx,ny,hx,hy,clx,cly,αx,αy,shift,regularization=false) where {T<:AbstractFloat}
    arraytype===Array || throw(ArgumentError("only CPU Array workspaces are supported"))
    clx in (1,2,3) && cly in (1,2,3) || throw(ArgumentError("boundary diagonal codes must be 1, 2 or 3"))
    isfinite(αx) && αx>0 && isfinite(αy) && αy>0 || throw(ArgumentError("diffusion coefficients must be positive"))
    regularization && throw(ArgumentError("use the zero-mean Neumann gauge instead of regularization"))
    TensorialOperator(T(αx)*K1(nx,hx,clx;T),T(αy)*K1(ny,hy,cly;T);shift=T(shift))
end
function _apply!(out,s::TensorialOperator,x)
    mul!(out,s.ax,x)
    mul!(out,x,s.ay,one(s.shift),one(s.shift))
    @. out += s.shift*x
    out
end
function solve!(x,b,s::TensorialOperator{T};reltol=_default_reltol(T),abstol=zero(T),maxiter=1000,store_history=true) where T
    size(x)==size(b)==s.shape || throw(DimensionMismatch("tensor arrays must have size $(s.shape)"))
    threshold=_prepare(x,b,s;reltol,abstol,maxiter)
    initial,history=_initial(x,b,s,threshold,store_history,maxiter)
    initial===nothing || return initial
    mul!(s.t1,transpose(s.mx),b)
    mul!(s.t2,s.t1,s.my)
    s.t2 .*= s.inverse_spectrum
    mul!(s.t1,s.mx,s.t2)
    mul!(x,s.t1,transpose(s.my))
    store_history && push!(history,zero(T))
    _result(x,b,s,1,history,threshold)
end
