"""
    GSSmoother(laplacian)

Red-black Gauss-Seidel smoother for a cell-centered rectangular grid. Boundary
contributions are included in the diagonal, including mixed boundary conditions.
"""
struct GSSmoother{T<:AbstractFloat}
    laplacian::Laplacian2D{T}
end
function _smooth!(x,b,s::GSSmoother)
    a=s.laplacian
    cx,cy=a.lpx.dxm2_,a.lpy.dxm2_
    nx,ny=size(a)
    @inbounds for color=0:1, j=1:ny
        for i=1+mod(color-j,2):2:nx
            neighbors=zero(eltype(x))
            i>1 && (neighbors+=cx*x[i-1,j])
            i<nx && (neighbors+=cx*x[i+1,j])
            j>1 && (neighbors+=cy*x[i,j-1])
            j<ny && (neighbors+=cy*x[i,j+1])
            x[i,j]=(b[i,j]+neighbors)/(a.lpx[i,i]+a.lpy[j,j])
        end
    end
    a.allneumann && _recenter!(x)
    x
end
