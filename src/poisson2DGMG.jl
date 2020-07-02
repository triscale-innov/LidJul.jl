# Adapted from C++ GMG Solver from Harald Köstler
# "Multigrid HowTo: A simple Multigrid solver in C++ in less than 200 lines of code"
#  https://www10.cs.fau.de/publications/reports/TechRep_2008-03.pdf

export PoissonGMG,solve!,boundary_coeff,treatboundary,fres,residual,restrict_residual,interpolate_correct,smooth,maxlevels
using SparseArrays
using LinearAlgebra
using BenchmarkTools
using LoopVectorization

struct PoissonGMG{SMOOTHER}
    Sol::Vector{Array{Float64,2}}
    Solb::Vector{Array{Float64,2}}
    RHS::Vector{Array{Float64,2}}
    invh2::Vector{Float64}
    smoothers::Vector{SMOOTHER}
    nlevels::Int
    nprae::Int
    npost::Int
    ncoarse::Int
    bc::NTuple{2,NTuple{2,BoundaryCondition}}
end

function maxlevels(n)
    try
       Int(log(n)/log(2)-1)
    catch
        error("nx=ny must be a power of two")
    end
end


function PoissonGMG(l2D,::Type{SMOOTHER};nlevels=-1,nprae=2,npost=1,ncoarse=20) where {SMOOTHER}
    Lx,Ly=l2D.Lx,l2D.Ly
    nx,ny=l2D.nx,l2D.ny
    nx!=ny && error("nx must be equal to ny")
    Lx!=Ly && error("Lx must be equal to Ly")
    bc=l2D.bc
    nlevels>maxlevels(nx) && error("nlevels too large")
    (nlevels==-1) && (nlevels=maxlevels(nx))
    # @show nlevels

    # add 1 ghost layer in each direction
    sizeadd=2
    Sol=Vector{Array{Float64,2}}(undef,nlevels)
    Solb=Vector{Array{Float64,2}}(undef,nlevels)
    RHS=Vector{Array{Float64,2}}(undef,nlevels)
    invh2=Vector{Float64}(undef,nlevels)
    smoothers=Vector{SMOOTHER}(undef,nlevels)
    nrows,ncols=nx,ny
    h2=1.0
    L=Lx
    for i=1:nlevels
        nlevels
        Sol[i]=zeros(nrows+sizeadd,ncols+sizeadd)
        Solb[i]=zeros(nrows+sizeadd,ncols+sizeadd)
        RHS[i]=zeros(nrows+sizeadd,ncols+sizeadd)
        h=L/(nrows)
        h2=h*h
        ih2=1/h2
        invh2[i]=ih2
        smoothers[i]=SMOOTHER(invh2[i],nrows,ncols,bc,(i==1))

        nrows=div(nrows,2)
        ncols=div(ncols,2)

    end
    PoissonGMG{SMOOTHER}(Sol,Solb,RHS,invh2,smoothers,nlevels,nprae,npost,ncoarse,bc)
end


function solve!(Xxy,Bxy,g::PoissonGMG)
    RHS,Sol,Solb,invh2=g.RHS,g.Sol,g.Solb,g.invh2
    nlevels=size(RHS,1)
    r1,s1=RHS[1],Sol[1]
    nrows,ncols=size(s1)

    @inbounds for j=2:ncols-1
        for i=2:nrows-1
            r1[i,j]=Bxy[i-1,j-1]
            s1[i,j]=Xxy[i-1,j-1]
        end
    end

    res_1=residual(1,g)
    residuals=img_solver(nlevels,res_1,g)
    (nx,ny)=size(Xxy)

    @inbounds for j=2:ncols-1
        for i=2:nrows-1
            Xxy[i-1,j-1]=s1[i,j]
        end
    end
    residuals
end



function img_solver(nlevels,res_1,g::PoissonGMG)
    res=res_1
    res_old=0.
    #no . of pre -, post -, and coarse smoothing steps
    nprae=g.nprae
    npost=g.npost
    ncoarse=g.ncoarse
    lev=1
    max_iters=20

    residuals=[res_1]
    for i=1:max_iters
        res_old=res
        VCycle(lev,g)
        res=residual(lev,g)
        push!(residuals,res)
        (res<1.e-9) && return residuals
        # @show lev,i,res,res_1/res,res/res_old
    end
    residuals
end


function smooth(lev,g::PoissonGMG)
    treatboundary(lev,g.Sol[lev],g)
    smooth(g.Sol[lev],g.RHS[lev],g.smoothers[lev])
    treatboundary(lev,g.Sol[lev],g)

end

function VCycle(lev,g)
    #solve problem on coarsest grid ...
    if lev==g.nlevels

        for i=1:g.ncoarse
            smooth(lev,g)
        end
    else
        #... or recursively do V - cycle
        #
        # do some presmoothing steps
        for i=1:g.nprae
            smooth(lev,g)
        end

        restrict_residual(lev,g)
        # initialize the coarse solution to zero
        fill!(g.Sol[lev+1],0.0);
        VCycle(lev+1,g)
        # interpolate error and correct fine solution
        interpolate_correct(lev+1,g)
        # do some postsmoothing steps
        for i=1:g.npost
            smooth(lev,g)
        end
    end
end

boundary_coeff(bc::BoundaryCondition) = (bc==dirichlet) ? -1.0 : 1.0


function treatboundary(lev,Sol,g)
    sl=Sol
    (nrows,ncols)=size(sl)
    bc=g.bc

    cleft=boundary_coeff(bc[1][1])
    cright=boundary_coeff(bc[1][2])
    cbottom=boundary_coeff(bc[2][1])
    ctop=boundary_coeff(bc[2][2])


    @inbounds for j=2:ncols-1
        sl[1,j]=cleft*sl[2,j]
        sl[nrows,j]=cright*sl[nrows-1,j]
    end
    @inbounds for i=2:nrows-1
        sl[i,1]=cbottom*sl[i,2]
        sl[i,ncols]=ctop*sl[i,ncols-1]
    end
    allneumann = bc[1][1]==bc[1][2]==bc[2][1]==bc[2][2]==neumann

end

# @inline fres(x,y,i,j,ih2)= @inbounds x[i,j]+ih2*(y[i+1,j]+y[i,j+1]+y[i,j-1]+y[i-1,j]-4y[i,j])-0.0001y[i,j]
@inline function fres(x,y,i,j,ih2,nx,ny)
    @inbounds x[i,j]+ih2*(y[i+1,j]+y[i,j+1]+y[i,j-1]+y[i-1,j]-4.0*y[i,j])
end


function residual(lev,g)
    sl=g.Sol[lev]
    rl=g.RHS[lev]
    ih2=g.invh2[lev]
    (nrows,ncols)=size(sl)
    res=0.0

    nx=nrows-1
    ny=ncols-1

    for j=2:ny
        @simd for i=2:nx
            rf=fres(rl,sl,i,j,ih2,nx,ny)
            res+=rf*rf
        end
    end
    sqrt(res)
end

# using LoopVectorization

function restrict_residual(lev,g)
    rl1=g.RHS[lev+1]
    sl1=g.Sol[lev+1]
    rl=g.RHS[lev]
    sl=g.Sol[lev]
    (nrows,ncols)=size(rl1)
    ih2=g.invh2[lev]

    nx=nrows-1
    ny=ncols-1
    fill!(rl1,0.0)
    fill!(sl1,0.0)


    @inline finer(i,j)=fres(rl,sl,i,j,ih2,nx,ny)

    @simd for j=2:ncols-1
        fj=2j-2

        for i=2:nrows-1
            fi=2i-2

             @inbounds rl1[i,j]=0.25*(finer(fi,fj)+finer(fi+1,fj)+finer(fi,fj+1)+finer(fi+1,fj+1))
            #  @inbounds rl1[i,j]=0.25*(finer(fi,fj)+finer(fi+1,fj)+finer(fi,fj+1)+finer(fi+1,fj+1))
        end
    end


end




function interpolate_correct(lev,g::PoissonGMG)
    uf=g.Sol[lev-1]
    uc=g.Sol[lev]
    (nrows,ncols)=size(uc)
    ncm1=ncols-1
    nrm1=nrows-1

    @inbounds for j=2:ncm1
        fj=2j-1
        @fastmath for i=2:nrm1
            fi=2i-1
             v=uc[i,j]
             uf[fi  ,fj  ]+=v
             uf[fi-1,fj  ]+=v
             uf[fi  ,fj-1]+=v
             uf[fi-1,fj-1]+=v
        end
    end

end



