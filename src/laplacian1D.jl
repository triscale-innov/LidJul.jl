using LinearAlgebra
export l1dpαI,multIxAy!,multAxIy!,compute_residual
export thomas_div!,laplace_div!
export l1dpαI_dirichlet,l1dpαI_neuman_left


struct Laplacian1D
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
        dx=L/n
        dxm2=1/(dx^2)
        d=2dxm2
        u=-1dxm2
        (bcright == neumann) ? ld=dxm2 : ld=3dxm2
        (bcleft == neumann) ? fd=dxm2 : fd=3dxm2
        new(dxm2,n,bcleft,bcright,fd,ld,d,u,L)
    end
end

function Base.checkbounds(a::Laplacian1D, i,j)
    ((i>1) && (j>1) && (i<=a.n_) && (j<=a.n_))
end
function Base.getindex(a::Laplacian1D, i,j)
    i==j==1 && return a.firstdiag_
    i==j==a.n_ && return a.lastdiag_
    @boundscheck checkbounds(a,i,j)
    i==j && return a.diag_
    i==(j+1) && return a.upperdiag_
    i==(j-1) && return a.upperdiag_
    0.
end
Base.size(Laplacian1D) = (Laplacian1D.n_,Laplacian1D.n_)
Base.size(Laplacian1D,i) = Laplacian1D.n_


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


