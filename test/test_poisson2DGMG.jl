using LidJul
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using Makie
using IterativeSolvers
using Preconditioners
using Random


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



include("measureplots.jl")


function testpoisson_GMG(n,bc)
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
    test_solver!(msm,PoissonGMG,(l2D,GSSmoother),splu_nocorr,b)

    test_solver!(msm,PoissonGMG_new,(l2D,GSSmoother),splu_nocorr,b)

  
    sname=string(n)*tostring(bc)

    @save "msm"*sname*".jld2" msm

    # measureplot(msm,sname)

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

function dump_measurments(msm,sname)
    header=keys(msm[1])[1:end]
    nrows=length(msm)
    ncols=length(header)
    table=Matrix{Any}(undef,nrows,ncols)
    iterations=Vector{Union{Nothing,Vector{Float64}}}(undef,nrows)
    for i=1:nrows
        for j=1:ncols
            table[i,j]=msm[i][header[j]]
        end
        table[i]=replace(table[i],"Poisson"=>"")
        table[i]=replace(table[i],"Preconditioner"=>"")
        table[i]=replace(table[i],"SparseCG"=>"PCG")
    end
     #Remove some measurements :
     removed=["GMG_CGSmoother"]
     table=table[filter(i->!(table[i,1] in removed),1:size(table,1)),:]
 
     #add Total Time and niter
     header=[header[1:3]...,"Total Time",header[4],"niters",header[5]]
     table=hcat(table[:,1:3],table[:,2].+table[:,3],table[:,4],niterations.(residuals(table[:,5])),table[:,5])
     #Sort
     sp=sortperm(table[:, 4])
     table=table[sp,:]
     # iterations=iterations[sp,:]
     nr,nc=size(table)
     println("all methods")
     pretty_table(table[:,1:6],collect(header[1:6]);formatters=ft_printf("%5.3E",2:5),alignment=:l)
end

function go()
    #Choose a power of two
    n=512
    #Choose boundary conditions
    # bc=(neumann,neumann,neumann,neumann)
    bc=(dirichlet,neumann,dirichlet,neumann)
    # bc=(dirichlet,dirichlet,dirichlet,dirichlet)

    sname=string(n)*tostring(bc)
    p,pref,pbj,s,b,msm=testpoisson_GMG(n,bc)
    @save "msm"*sname*".jld2" msm

    dump_measurments(msm,sname)
    @show msm
    msm



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

msm=go()
