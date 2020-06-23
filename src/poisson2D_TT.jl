export PoissonTTSolver,solve!
using LinearAlgebra

struct PoissonTTSolver
    Mx::Array{Float64,2}
    My::Array{Float64,2}
    Mxm1::Array{Float64,2}
    Mym1::Array{Float64,2}
    Dx::Array{Float64,1}
    Dy::Array{Float64,1}
    t1::Array{Float64,2}
    t2::Array{Float64,2}

    function PoissonTTSolver(lp2D)
        # Lx=l1dpαI_dirichlet(nx,dx,0.0)
        Lx=SymTridiagonal(lp2D.lpx)
        nx=size(Lx,1)
        Ly=SymTridiagonal(lp2D.lpy)
        ny=size(Ly,1)

        # Ly=l1dpαI_dirichlet(ny,dy,0.0)

        Ex=eigen(Lx)
        Mx=Ex.vectors
        Dx=Ex.values
        Mxm1=inv(Mx)

        Ey=eigen(Ly)
        My=Ey.vectors
        Dy=Ey.values
        Mym1=inv(My)
        t1=zeros(nx,ny)
        t2=zeros(nx,ny)
        new(Mx,My,Mxm1,Mym1,Dx,Dy,t1,t2)
    end
end

function solve!(Xxy,Bxy,ps::PoissonTTSolver)
    (Mx,My,Mxm1,Mym1,Dx,Dy,t1,t2)=(ps.Mx,ps.My,ps.Mxm1,ps.Mym1,ps.Dx,ps.Dy,ps.t1,ps.t2)
    (nx,ny)=size(Xxy)
    @assert size(Xxy)==size(Bxy)==(size(Mx,1),size(My,2))

    LinearAlgebra.mul!(t1,Mym1,Bxy)
    LinearAlgebra.mul!(t2,Mxm1,transpose(t1))

    for j=1:ny
        @simd for i=1:nx
            @inbounds t2[i,j]/=(Dx[i]+Dy[j])
        end
    end

    LinearAlgebra.mul!(t1,My,t2)
    LinearAlgebra.mul!(Xxy,Mx,transpose(t1))
    nothing
    # Xxy
end
