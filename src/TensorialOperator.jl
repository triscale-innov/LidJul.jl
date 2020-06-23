#Solves Poisson or Helmoltz equation with separable tensorial operator.
# $T_{xy} X_{xy} = B_{xy}$
# Let $T_{xy} = Tx⊗Iy+Ix⊗Ty+λIx⊗Iy$
# via tensorial expression of $T^{-1}_{xy}
# Let $M_x$ and $M_y$ be the diagonalization matrices of $T_x$ and $T_y$ :
# $Tx=M_x I_x M_x^{-1}$ and $Tx=M_y I_x M_y^{-1}$
# ..
# See for example : Plagne, Laurent, and Jean-Yves Berthou. 
# "Tensorial basis spline collocation method for Poisson's equation." 
# Journal of Computational Physics 157.2 (2000): 419-440.

using LinearAlgebra
export TensorialOperator

struct TridiagOperator{A2D,A1D}
   m::A2D
   mi::A2D
   d::A1D
end

function TridiagOperator(A,T,n,h,cl,α,regularization=false)
   k1=α*K1(n,h,cl)
   if regularization
      k1[1,1]=3/2*k1[1,1];
   end
   tri=SymTridiagonal(collect(diag(k1,0)),collect(diag(k1,1)))
   e=eigen(tri)
   m=e.vectors
   d=e.values
   mi=inv(m)
   TridiagOperator{A{T,2},A{T,1}}(m,mi,d)
end


# Tx⊗Iy+Ix⊗Ty+λIx⊗Iy
struct TensorialOperator{A2D,A1D}
   Tx::TridiagOperator{A2D,A1D}
   Ty::TridiagOperator{A2D,A1D}
   λ::Float64
   nx::Int
   ny::Int
   txy::A2D
   tyx::A2D
   tyx2::A2D
   Dxy::A2D
end


function TensorialOperator(A,T,nx,ny,hx,hy,clx,cly,αx,αy,λ,regularization=false)
   Tx=TridiagOperator(A,T,nx,hx,clx,αx,regularization)
   Ty=TridiagOperator(A,T,ny,hy,cly,αy,regularization)
   Dxy=zeros(nx,ny)

   txd=Array(Tx.d)
   tyd=Array(Ty.d)


   Dxy .= T(1.0) ./ (T(λ) .+ kron(txd,ones(ny)') .+ kron(ones(nx),tyd'))

   TensorialOperator{A{T,2},A{T,1}}(Tx,Ty,λ,nx,ny,zeros(nx,ny),zeros(ny,nx),zeros(ny,nx),Dxy)
end


function solve!(Rxy,Bxy,Txy::TensorialOperator)

   Tx,Ty=Txy.Tx,Txy.Ty
   Mx,Mxm1,Dx=Tx.m,Tx.mi,Tx.d
   My,Mym1,Dy=Ty.m,Ty.mi,Ty.d
   txy,tyx,tyx2=Txy.txy,Txy.tyx,Txy.tyx2
   nx,ny,λ=Txy.nx,Txy.ny,Txy.λ

   LinearAlgebra.mul!(txy,Mxm1,Bxy)
   LinearAlgebra.mul!(tyx,Mym1,transpose(txy))

   tyx .*= (Txy.Dxy)'

   LinearAlgebra.mul!(tyx2,My,tyx)
   LinearAlgebra.mul!(Rxy,Mx,transpose(tyx2))

end

function speye(n)
   SparseMatrixCSC(1.0*SparseArrays.I, n, n)
end


function K1(n,h,a11)
# a11: Neumann=1, Dirichlet=2, Dirichlet mid=3;
   h2m1=1/(h^2)
   a=spdiagm(-1 => -ones(n-1), 0 => 2ones(n), 1 => -ones(n-1))
   a[1,1]=a11
   a[n,n]=a11
   a.*=h2m1
   a
end

function K2(n,h,a11,ann)
# a11: Neumann=1, Dirichlet=2, Dirichlet mid=3;
   h2m1=1/(h^2)
   a=spdiagm(-1 => -ones(n-1), 0 => 2ones(n), 1 => -ones(n-1))
   a[1,1]=a11
   a[n,n]=ann
   a.*=h2m1
   a
end
