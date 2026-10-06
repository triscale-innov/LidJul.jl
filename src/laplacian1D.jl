"""
    Laplacian1D(n, length, left, right; T=Float64)

Cell-centered negative second derivative on `n >= 2` cells, with homogeneous
boundary conditions. `T` must be `Float32` or `Float64`. Implements the standard
`AbstractMatrix` interface and conversion to `SymTridiagonal`.
"""
struct Laplacian1D{T<:AbstractFloat} <: AbstractMatrix{T}
    dxm2_::T
    n_::Int
    bcr_::BoundaryCondition
    bcl_::BoundaryCondition
    firstdiag_::T
    lastdiag_::T
    diag_::T
    upperdiag_::T
    L_::T
end

function Laplacian1D(n::Integer,L::Real,left::BoundaryCondition,right::BoundaryCondition;
                     T::Type{<:AbstractFloat}=Float64)
    T in (Float32,Float64) || throw(ArgumentError("supported types are Float32 and Float64"))
    n >= 2 || throw(ArgumentError("n must be at least 2"))
    isfinite(L) && L > 0 || throw(ArgumentError("length must be finite and positive"))
    h2 = inv((T(L)/n)^2)
    Laplacian1D{T}(h2,n,right,left,(left==neumann ? one(T) : T(3))*h2,
                  (right==neumann ? one(T) : T(3))*h2,T(2)*h2,-h2,T(L))
end

Base.size(a::Laplacian1D) = (a.n_,a.n_)
function Base.getindex(a::Laplacian1D{T},i::Int,j::Int) where T
    @boundscheck checkbounds(a,i,j)
    i==j==1 && return a.firstdiag_
    i==j==a.n_ && return a.lastdiag_
    i==j && return a.diag_
    abs(i-j)==1 && return a.upperdiag_
    zero(T)
end
function LinearAlgebra.SymTridiagonal(a::Laplacian1D)
    d=fill(a.diag_,a.n_)
    d[1],d[end]=a.firstdiag_,a.lastdiag_
    SymTridiagonal(d,fill(a.upperdiag_,a.n_-1))
end
