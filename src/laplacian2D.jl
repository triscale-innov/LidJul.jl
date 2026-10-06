"""
    Laplacian2D(nx, ny, Lx, Ly, left, right, bottom, top; T=Float64)

Cell-centered negative Laplacian on a rectangle. Boundary values are homogeneous
and ordered left, right, bottom, top. Supports `Float32` and `Float64`. A pure
Neumann operator has the constants as its null space.
"""
struct Laplacian2D{T<:AbstractFloat}
    lpx::Laplacian1D{T}
    lpy::Laplacian1D{T}
    nx::Int
    ny::Int
    Lx::T
    Ly::T
    bc::NTuple{2,NTuple{2,BoundaryCondition}}
    allneumann::Bool
end
function Laplacian2D(nx::Integer,ny::Integer,Lx,Ly,left,right,bottom,top;
                     T::Type{<:AbstractFloat}=Float64)
    ax=Laplacian1D(nx,Lx,left,right;T)
    ay=Laplacian1D(ny,Ly,bottom,top;T)
    Laplacian2D{T}(ax,ay,nx,ny,T(Lx),T(Ly),((left,right),(bottom,top)),
                   all(==(neumann),(left,right,bottom,top)))
end
Base.size(a::Laplacian2D)=(a.nx,a.ny)
function SparseArrays.sparse(a::Laplacian2D{T}) where T
    ax=sparse(SymTridiagonal(a.lpx))
    ay=sparse(SymTridiagonal(a.lpy))
    kron(spdiagm(0=>ones(T,a.ny)),ax)+kron(ay,spdiagm(0=>ones(T,a.nx)))
end

"""
    sparse_corr(laplacian)

Return a sparse matrix with a diagonal anchor for pure Neumann problems.
This helper modifies the operator; use `sparse(laplacian)` to check physical
residuals. Prefer solver constructors taking `laplacian`, which anchor internally
and return a common zero-mean gauge without changing residual measurements.
"""
function sparse_corr(a::Laplacian2D)
    matrix=sparse(a)
    a.allneumann && (matrix[1,1] *= 3/2)
    matrix
end
