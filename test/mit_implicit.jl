using SparseArrays
using LinearAlgebra
using Makie
using LidJul
using DataStructures #for DefaultDict
using PrettyTables
using StaticArrays
using Interpolations
using SIMD




#MIT18086_NAVIERSTOKES
#    Solves the incompressible Navier-Stokes equations in a
#    rectangular domain with prescribed velocities along the
#    boundary. The solution method is finite differencing on
#    a staggered grid with implicit diffusion and a Chorin
#    projection method for the pressure.
#    Visualization is done by a colormap-isoline plot for
#    pressure and normalized quiver and streamline plot for
#    the velocity field.
#    The standard setup solves a lid driven cavity problem.
# 07/2007 by Benjamin Seibold
#            http://www-math.mit.edu/~seibold/
# Feel free to modify for teaching and learning.
const   ARRAY=Array
const   REAL=Float64
function mit18086_navierstokes()
   # scene=Scene(resolution=(1000,1000),scale=false)
   # display(scene)
   #-----------------------------------------------------------------------
   # CuArrays.allowscalar(false)

   mte=DefaultDict{String,Float64}(0.0)
   mte["initialization"]+=@elapsed   begin

   Re = 1.e4     # Reynolds number
   dt = 0.5e-2;    # time step
   tf = 50e-0;    # final time
   lx = 1.0;       # width of box
   ly = 1.0;       # height of box
   nx = 128 ;      # number of x-gridpoints
   ny = 128 ;      # number of y-gridpoints
   #nx = 41;
   #ny = 41;
   nsteps = 120;  # number of steps with graphic output
   #-----------------------------------------------------------------------
   nt = ceil(tf/dt); #dt = tf/nt;
   @show nt,dt


   x = range(0,stop=lx,length=nx+1); hx = lx/nx;
   y = range(0,stop=ly,length=ny+1); hy = ly/ny;

   println("initialization")

   L2dp=Laplacian2D(nx,ny,1,1,neumann,neumann,neumann,neumann)
   gmgp=PoissonGMG(L2dp,GSSmoother)
   Tlp=TensorialOperator(ARRAY,REAL,nx,ny,hx,hy,3,3,1.0,1.0,1.e-5,false)
   Tlu=TensorialOperator(ARRAY,REAL,nx-1,ny,hx,hy,2,3,dt/Re,dt/Re,1)
   Tlv=TensorialOperator(ARRAY,REAL,nx,ny-1,hx,hy,3,2,dt/Re,dt/Re,1)
   Tlq=TensorialOperator(ARRAY,REAL,nx-1,ny-1,hx,hy,2,2,1,1,0)

   Ue=zeros(nx+2,ny+2)
   Ve=zeros(nx+2,ny+2)

   uN = collect(x*0.0 .+ 1.0);  vN=zeros(nx) #vN = avg(x)*0;
   uS = collect(x*0.0);  vS=zeros(nx) #vS = avg(x)*0;
   uW = zeros(ny); vW = collect(y*0.0);
   uE = zeros(ny); vE = collect(y*0.0);

   Ue[1,2:ny+1].=uW
   Ue[nx+1,2:ny+1].=uE
   Ue[1:nx+1,1].=2.0.*uS.-Ue[1:nx+1,2]
   Ue[1:nx+1,end].=2.0.*uN.-Ue[1:nx+1,end-1]

   Ve[2:end-1,1].=vS
   Ve[2:end-1,end].=vN
   Ve[1,1:ny+1].=2.0.*vW.-Ve[2,1:ny+1]
   Ve[end,1:ny+1].=2.0.*vE.-Ve[end-1,1:ny+1]


   U=view(Ue,2:nx,2:ny+1)
   V=view(Ve,2:nx+1,2:ny)

   UVx=zeros(nx,ny)
   UVy=zeros(nx,ny)
   UVx2=zeros(nx,ny)
   UVy2=zeros(nx,ny)

   vUVy=view(UVy,2:nx,1:ny)
   vUVx=view(UVx,1:nx,2:ny)
   vUVx2=view(UVx2,1:nx-1,1:ny)
   vUVy2=view(UVy2,1:nx,1:ny-1)

   Bx = zeros(nx+1,ny); By = zeros(nx,ny+1); Bxy=zeros(nx,ny)

   Bx[1,:] .= uW
   Bx[end,:] .= uE
   By[:,1] .= vS
   By[:,end] .= vN

   bU=zeros(nx-1,ny)
   bV=zeros(nx,ny-1)
   Qbxy=zeros(nx-1,ny-1)
   P=zeros(nx,ny) ;  Q = zeros(nx-1,ny-1)

   end

   mte["init Makie"]+=@elapsed   begin

   scene,qn,un,vn,uvn=plotflow_makie(nx,ny,hx,hy,x,y,U,V,Ue,Ve,P,Q,Tlq)

   end
   display(scene)
   record(scene,"test.gif",framerate=6) do io

   hxm1=1/hx
   hym1=1/hy

      @inbounds for k = 1:nt

      # treat nonlinear terms
      mte["gamma"]+=@elapsed begin
         # mu=maxabs(U)
         # mv=maxabs(V)
         mu=vmaxabs(Ue,SIMD.Vec{16,Float64})
         mv=vmaxabs(Ve,SIMD.Vec{16,Float64})

         gamma = min(1.2*dt*max(mu/hx,mv/hy),1.0);
      end

      mte["Ue boundary"]+=@elapsed begin
         @simd for i=1:nx+1
            Ue[i,1]=2.0uS[i]-Ue[i,2]
            Ue[i,end]=2.0uN[i]-Ue[i,end-1]
         end
      end

      mte["Ve boundary"]+=@elapsed begin
         @simd for j=1:ny+1
            Ve[1,j]=2.0vW[j]-Ve[2,j]
            Ve[end,j]=2.0vE[j]-Ve[end-1,j]
         end
      end

      compute_derivative!(UVx, UVx2, UVy, UVy2,Ue,Ve,nx,ny,gamma,hxm1,hym1,mte)



      mte["bu"]+=@elapsed begin
         for j=1:ny
            for i=1:nx-1
               bU[i,j]=U[i,j]-dt*(vUVy[i,j]+vUVx2[i,j]);
            end
         end
         border=(dt/Re)*2/(hx^2)
         for i=1:nx-1
            bU[i,ny] +=  border
         end
      end

      mte["bv"]+=@elapsed   begin
         for j=1:ny-1
            for i=1:nx
               bV[i,j]=V[i,j]-dt*(vUVx[i,j]+vUVy2[i,j]);
            end
         end

      end
      mte["solve TT"]+=@elapsed solve!(U,bU,Tlu)
      mte["solve TT"]+=@elapsed solve!(V,bV,Tlv)


         # pressure correction
      mte["Bxy"]+=@elapsed   begin
         j=1
         i=1
         Bxy[i,j]=-((U[i,j]-Bx[i,j])*hxm1+(V[i,j]-By[i,j])*hym1)
         for i=2:nx
            Bxy[i,j]=-((U[i,j]-U[i-1,j])*hxm1+(V[i,j]-By[i,j])*hym1)
         end
         i=1
         for j=2:ny
               Bxy[i,j]=-((U[i,j]-Bx[i,j])*hxm1+(V[i,j]-V[i,j-1])*hym1)
         end
         for j=2:ny
            for i=2:nx

               Bxy[i,j]=-((U[i,j]-U[i-1,j])*hxm1+(V[i,j]-V[i,j-1])*hym1)
            end
         end
      end

      mte["solveMG"]+=@elapsed LidJul.solve!(P,Bxy,gmgp)

      # @time begin
      mte["U V"]+=@elapsed begin
         @simd for j=1:ny
            for i=1:nx-1

               U[i,j] -= (P[i+1,j]-P[i,j])*hxm1
            end
         end
         @inbounds for j=1:ny-1
            for i=1:nx
               V[i,j] -= (P[i,j+1]-P[i,j])*hym1
            end
         end
      end



      # visualization
      mte["Makie Frames"]+=@elapsed begin
      if floor(25*k/nt)>floor(25*(k-1)/nt)
         print('.')
      end
      #
      if k==1 || floor(nsteps*k/nt)>floor(nsteps*(k-1)/nt)
         mte["solve Qbxy"]+=@elapsed begin
         @inbounds for j=1:ny-1
            for i=1:nx-1
               Qbxy[i,j]=(U[i,j+1]-U[i,j])*hym1-(V[i+1,j]-V[i,j])*hxm1
            end
         end
         end
         mte["solve TTQ"]+=@elapsed solve!(Q,Qbxy,Tlq)

  
         Qb=zeros(nx+1,ny+1)
         Qb[2:end-1,2:end-1].=Q
         qn[] = copy!(to_value(qn),Qb)
         uvn[] = copy!(to_value(uvn),[Ue,Ve])
         recordframe!(io) # record a new frame

      end
      end
   end
end
    sort_and_dump(mte)
   return
end


# function make_grad_qn(vq)

#    function grad_qn(x,y)
#       # @show x,y
#       nxq,nyq=size(vq)
#       # @show nxq,nyq
#       xi=1+x*(nxq-1)
#       yi=1+y*(nyq-1)
#       # @show xi,yi
#        f_qn=interpolate(vq,BSpline(Linear()))
#        Point2f0(Interpolations.gradient(f_qn,xi,yi))
#    end
#    return grad_qn
# end

function make_velocity(uv)
      f_vx=interpolate(uv[1],BSpline(Quadratic(Reflect(OnCell()))))
      f_vy=interpolate(uv[2],BSpline(Quadratic(Reflect(OnCell()))))
   function velocity(x,y)

      nx,ny=size(uv[1])
      xi=1+x*(nx-1)
      yi=1+y*(ny-1)

      Point2f0(f_vx(xi,yi),f_vy(xi,yi))
   end
   return velocity
end


function plotflow_makie(nx,ny,hx,hy,x,y,U,V,Ue,Ve,P,Q,Tlq)
   scene=Scene(resolution=(300,300),scale=false)
   limits = FRect(0.2,0.2, 0.72, 0.72)


   sp=2
   stp=1
   endp=2

   qn=Node(rand(nx+1,ny+1))

   uvn=Node([rand(size(Ue)...),rand(size(Ve)...)])
   un=Node(rand(size(Ue)...))
   vn=Node(rand(size(Ve)...))
   # lift(a->println(size(a),norm(a)),qn)
   as=2


   nlevels=10
   contour!(scene,x,y,lift(a->a,qn),linewidth=1,levels=200,fillrange=false,limits=limits,colormap = :plasma, axis = (showticks = false, showaxis = false,), alpha=0.95)
   streamplot!(scene,lift(a->make_velocity(to_value(a)),uvn),x,y, colormap = :magma, arrow_size=0.02, gridsize=(20,20))

   axis = scene[Axis]
   axis[:names][:axisnames] = (" ", " ")

   scene,qn,un,vn,uvn
end


function maxabs(U)
   minu=typemax(eltype(U))
   maxu=typemin(eltype(U))
   for uij in U
      minu > uij && (minu=uij)
      maxu < uij && (maxu=uij)
   end
   max(abs(minu),abs(maxu))
end

# function vmaxabs(xin, ::Type{SIMD.Vec{N,T}}=SIMD.Vec{4,Float64}) where {N, T}
#LP: needs more tests
function vmaxabs(xin, ::Type{SIMD.Vec{N,T}}) where {N, T}
   l=length(xin)
   xs=reshape(xin,(l,))
   lane = VecRange{N}(0)

   xmv= SIMD.Vec{N,T}(typemin(T))
   nb=div(l,N)
   is=1
   @inbounds for ib in 1:nb
      xmv=max(xmv,abs(xs[lane + is]))
      is+=N
   end
   xm=maximum(xmv)
   @inbounds for i=is:l
      xm=max(xm,abs(xs[is]))
   end
   xm
end



function sort_and_dump(dict)
   s=reverse(sort(collect(dict), by=x->x[2]))
   println("")
   t=collect(x[2] for x in s)
   # @show t
   total=sum(t)
   @show total
   for si in s
      si[2]>0.001total && println(si)
   end
end



const BS=4

#LP: The ugly part which should be enhanced via V. Churavy techniques 
function compute_derivative!(UVx, UVx2, UVy, UVy2,Ue,Ve,nx,ny,gamma,hxm1,hym1,mte)


      njb=div(ny,BS)
      nib=div(nx,BS)

   mte["UVx static"]+=@elapsed begin
      for jb=1:njb
         jd=BS*(jb-1)

         @inbounds for ib=1:nib
            id=BS*(ib-1)
            # @show id,jd

            # @inbounds ue00=SMatrix{BS,BS,Float64}((Ue[id+1
            # [Ue[i,j] for i=ia+1:ia+BS, j=ja+1:ja+BS]
            ue00=@inbounds @SMatrix [Ue[ic+id+0,jc+jd+0] for ic=1:BS,jc=1:BS]
            ue10=@inbounds @SMatrix [Ue[ic+id+1,jc+jd+0] for ic=1:BS,jc=1:BS]
            ue01=@inbounds @SMatrix [Ue[ic+id+0,jc+jd+1] for ic=1:BS,jc=1:BS]
            ue11=@inbounds @SMatrix [Ue[ic+id+1,jc+jd+1] for ic=1:BS,jc=1:BS]
            ue21=@inbounds @SMatrix [Ue[ic+id+2,jc+jd+1] for ic=1:BS,jc=1:BS]
            ue02=@inbounds @SMatrix [Ue[ic+id+0,jc+jd+2] for ic=1:BS,jc=1:BS]


            ve00=@inbounds @SMatrix [Ve[ic+id+0,jc+jd+0] for ic=1:BS,jc=1:BS]
            ve10=@inbounds @SMatrix [Ve[ic+id+1,jc+jd+0] for ic=1:BS,jc=1:BS]
            ve20=@inbounds @SMatrix [Ve[ic+id+2,jc+jd+0] for ic=1:BS,jc=1:BS]
            ve01=@inbounds @SMatrix [Ve[ic+id+0,jc+jd+1] for ic=1:BS,jc=1:BS]
            ve11=@inbounds @SMatrix [Ve[ic+id+1,jc+jd+1] for ic=1:BS,jc=1:BS]
            ve12=@inbounds @SMatrix [Ve[ic+id+1,jc+jd+2] for ic=1:BS,jc=1:BS]

            uay00=0.5.*(ue01.+ue00)
            udy00=0.5.*(ue01.-ue00)
            uay01=0.5.*(ue02.+ue01)
            udy01=0.5.*(ue02.-ue01)
            uay10=0.5.*(ue11.+ue10)
            uax00=0.5.*(ue11.+ue01)
            udx00=0.5.*(ue11.-ue01)
            uax10=0.5.*(ue21.+ue11)
            udx10=0.5.*(ue21.-ue11)
         #
            vax00=0.5.*(ve10.+ve00)
            vdx00=0.5.*(ve10.-ve00)
            vax10=0.5.*(ve20.+ve10)
            vdx10=0.5.*(ve20.-ve10)
            vax01=0.5.*(ve11.+ve01)
            vay10=0.5.*(ve11.+ve10)
            vdy10=0.5.*(ve11.-ve10)
            vay11=0.5.*(ve12.+ve11)
            vdy11=0.5.*(ve12.-ve11)
            #
            uva100=uay00.*vax00.-gamma.*abs.(uay00).*vdx00
            uva110=uay10.*vax10.-gamma.*abs.(uay10).*vdx10
            uvx=hxm1.*(uva110.-uva100)

            uva2200=vay10.*vay10.-gamma.*abs.(vay10).*vdy10
            uva2201=vay11.*vay11.-gamma.*abs.(vay11).*vdy11
            uvy2=hym1.*(uva2201.-uva2200)

            uva200=uay00.*vax00.-gamma.*abs.(vax00).*udy00
            uva201=uay01.*vax01.-gamma.*abs.(vax01).*udy01
            uvy=hym1.*(uva201.-uva200)

            uva1200=uax00.*uax00.-gamma.*abs.(uax00).*udx00
            uva1210=uax10.*uax10.-gamma.*abs.(uax10).*udx10
            uvx2=hxm1.*(uva1210.-uva1200)


            # # #
            for jc=1:BS

               @simd for ic=1:BS
                  @inbounds UVx[id+ic,jd+jc]=uvx[ic,jc]
               end
               @simd for ic=1:BS
                  @inbounds UVy[id+ic,jd+jc]=uvy[ic,jc]
               end
               @simd for ic=1:BS
                  @inbounds UVx2[id+ic,jd+jc]=uvx2[ic,jc]
               end
               @simd for ic=1:BS
                  @inbounds UVy2[id+ic,jd+jc]=uvy2[ic,jc]
               end
            end


         end
      end
   end

end


@time mit18086_navierstokes()
