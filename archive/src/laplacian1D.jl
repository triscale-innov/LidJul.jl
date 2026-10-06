using LinearAlgebra
export Laplacian1D


struct Laplacian1D <: AbstractMatrix{Float64}
    dxm2_::Float64
    n_::Int
    bcr_::BoundaryCondition
    bcl_::BoundaryCondition
    firstdiag_::Float64
    lastdiag_::Float64
    diag_::Float64
    upperdiag_::Float64
    L_::Float64
    function Laplacian1D(n,L,bcleft::BoundaryCondition,bcright::BoundaryCondition)
        n >= 2 || throw(ArgumentError("n must be at least 2"))
        isfinite(L) && L > 0 || throw(ArgumentError("L must be finite and positive"))
        dx=L/n
        dxm2=1/(dx^2)
        d=2dxm2
        u=-1dxm2
        (bcright == neumann) ? ld=dxm2 : ld=3dxm2
        (bcleft == neumann) ? fd=dxm2 : fd=3dxm2
        new(dxm2,n,bcright,bcleft,fd,ld,d,u,L)
    end
end

function Base.getindex(a::Laplacian1D, i,j)
    @boundscheck checkbounds(a,i,j)
    i==j==1 && return a.firstdiag_
    i==j==a.n_ && return a.lastdiag_
    i==j && return a.diag_
    i==(j+1) && return a.upperdiag_
    i==(j-1) && return a.upperdiag_
    0.
end
Base.size(a::Laplacian1D) = (a.n_,a.n_)


function LinearAlgebra.SymTridiagonal(a::Laplacian1D)
    n=a.n_
    D=ones(n)
    D .*= a.diag_
    D[1]=a.firstdiag_
    D[end]=a.lastdiag_
    U=ones(n-1)
    U .*= a.upperdiag_
    SymTridiagonal(D,U)
end

