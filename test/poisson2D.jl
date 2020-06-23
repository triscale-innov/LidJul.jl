using LinearAlgebra
using SparseArrays
using BenchmarkTools
using Makie
using IterativeSolvers
using Preconditioners
using Random
using JLD2
using LidJul


function addmeasurements(msm,solvername,ti,ts,res,iterations=[])
    push!(msm,(solver=solvername,init_time=ti,solver_time=ts,residual=res,iterations=iterations))
end

function resnorm(splu,pref,b)
    ax=similar(b)
    LidJul.mul!(ax,splu,pref)
    norm(ax-b)
end

function test_solver!(msm,Solver,solver_args,splu,b)
    p=zeros(size(b))
    init_time=@elapsed solver=Solver(solver_args...)
    init_time=@elapsed solver=Solver(solver_args...)
    Random.seed!(1234)
    prand=rand(length(p))
    iterations=begin copyto!(p,prand) ; solve!(p,b,solver)  end
    solve_time=@elapsed  begin copyto!(p,prand) ; solve!(p,b,solver) end
    sname=string(Solver)
    length(solver_args)>1 && (sname=sname*"_"*string(solver_args[2]))

    addmeasurements(msm,sname,init_time,solve_time,resnorm(splu,p,b),iterations)
    p
end

function precoconstructor(A,preco)
    if preco==CholeskyPreconditioner
        return preco(A,4)
    elseif preco===nothing
        return IterativeSolvers.Identity()
    else
        return preco(A)
    end
end


function test_pcg(msm,preco,A,splu,Bxy)
    b=view(Bxy,1:length(Bxy))
    p=zeros(size(b))
    # x=view(Xxy,1:length(Xxy))
    sname="PCG"

    init_time=@elapsed pc=precoconstructor(A,preco)
    init_time=@elapsed pc=precoconstructor(A,preco)
    Random.seed!(1234)
    prand=rand(length(p))
    p,iterations = begin copyto!(p,prand) ; IterativeSolvers.cg!(p,A, b, Pl=pc;log=true,tol = 1e-10,maxiter=2000) end
    solve_time = @elapsed begin copyto!(p,prand) ; IterativeSolvers.cg!(p,A, b, Pl=pc;log=true,tol = 1e-10,maxiter=2000) end
    spreco=string(preco)[1:min(15,length(string(preco)))]
    sname="PCG"*spreco
    addmeasurements(msm,sname,init_time,solve_time,resnorm(splu,p,b),iterations)
    nothing
end
function test_stationary(msm,stm,A,splu,Bxy)
    spiter=PoissonSparseIterative(A)
    init_time=@elapsed PoissonSparseIterative(A)
    Xxy=similar(Bxy)
    Random.seed!(1234)
    Xxyrand=rand(size(Xxy))
    p,iterations = begin copyto!(Xxy,Xxyrand) ; solve!(Xxy,Bxy,spiter,stm) end
    solve_time = @elapsed begin copyto!(Xxy,Xxyrand) ; solve!(Xxy,Bxy,spiter,stm) end
    sname=string(stm)
    addmeasurements(msm,sname,init_time,solve_time,resnorm(splu,Xxy,Xxy),iterations)
    nothing
end

include("measureplots.jl")


function testpoisson(n,bc)
    msm=Vector{NamedTuple{(:solver,:init_time,:solver_time,:residual,:iterations),Tuple{String,Float64,Float64,Float64,Any}}}()

    p=ones(n,n)
    pref=ones(n,n)
    b=zeros(n,n)
    #LP: Create a Poisson's RHS compatible with a possible full Neumann boundary conditions
    #See bellow
    for i=1:n
        for j=1:n
            b[i,j]=sin(3π*(i-1)/(n-1))*sin(6π*(j-1)/(n-1))
        end
    end

    L=1.0
    dx=L/n
    dy=L/n
    @show dx,dy


    l2D=Laplacian2D(n,n,L,L,bc...)
    if bc==(neumann,neumann,neumann,neumann)
        #LP: for all neumann conditions the laplacian equation has no
        #solution if the rhs b does not belong to the range of A.
        #This impose the integral of b to be zero on the domain and
        # the sum of b on the mesh to be also zero...
        #See Mc Cormick and other for more details.
        @show sum(b)
        @assert(sum(b)<1.e-8)

        #Note that if this compatibility condition is ensured, and
        # x is a solution, then x+Cte is also a solution.
    end


    @time s2D=sparse_corr(l2D)
    splu,tlu=PoissonSparseLU(s2D)
    ts=@elapsed solve!(pref,b,splu)

    @time s2D_nocorr=SparseArrays.sparse(l2D)
    splu_nocorr,tlu_nocorr=PoissonSparseLU(s2D_nocorr)
    stationary_methods=["jacobi","gauss_seidel","sor","ssor"]
    for stm in stationary_methods
        test_stationary(msm,stm,s2D,splu,b)
    end

    test_pcg(msm,Preconditioners.DiagonalPreconditioner,s2D,splu,b)
    test_pcg(msm,nothing,s2D,splu,b)
    test_pcg(msm,Preconditioners.AMGPreconditioner{SmoothedAggregation},s2D,splu,b)
    test_pcg(msm,Preconditioners.AMGPreconditioner{RugeStuben},s2D,splu,b)
    test_solver!(msm,PoissonSparseAMG,(s2D,),splu,b)


    test_solver!(msm,PoissonTTSolver,(l2D,),splu,b)

    addmeasurements(msm,string(PoissonSparseLU),tlu,ts,resnorm(splu,pref,b))

    test_solver!(msm,PoissonGMG,(l2D,GSSmoother),splu_nocorr,b)

    test_solver!(msm,PoissonSparseCGILU,(s2D,),splu,b)

    sname=string(n)*tostring(bc)

    @save "msm"*sname*".jld2" msm

    measureplot(msm,sname)

    return p,pref,p,splu,b,msm

end



function tostring(bcs)
     r=string()
     for bc in bcs
         bc==dirichlet && (r*="D")
         bc==neumann && (r*="N")
    end
    r
end


function go()
    #Choose a power of two
    n=128
    #Choose boundary conditions 
    # bc=(neumann,neumann,neumann,neumann)
    bc=(dirichlet,neumann,dirichlet,neumann)
    # bc=(dirichlet,dirichlet,dirichlet,dirichlet)


    p,pref,pbj,s,b,msm=testpoisson(n,bc)

   #*** Makie Plots
    x = range(0,1,length=n)
    y = range(0,1,length=n)

    p=pref

    pmax=maximum(p)-minimum(p)
  
    p./=pmax

    s=Makie.surface(x,y,p)
    xm, ym, zm = minimum(scene_limits(s))
    Makie.contour!(s,x,y,p, levels = 15, linewidth = 2, transformation = (:xy, zm))
    Makie.wireframe!(s,x,y,p, overdraw = true, transparency = true, color = (:black, 0.1))
    display(AbstractPlotting.PlotDisplay(), s)
    resize!(s,(1600,800))
    Makie.save("makie"*string(n)*tostring(bc)*".png", s)
    nothing
end

go()
