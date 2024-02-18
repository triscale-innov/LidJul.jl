using LidJul
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using GLMakie
using IterativeSolvers
using Preconditioners
using Random


function save_vector!(io,x)
    write(io,length(x))
    for xi ∈ x
        write(io,xi)
    end
end

function read_vector!(io,T::DataType)
    x = Vector{T}()
    n = read(io,Int64)
    for i ∈ 1:n
        xi=read(io,T)
        push!(x,xi)
    end
    x
end

function save_sparse_matrix!(io,spA)
    write(io,spA.m)
    write(io,spA.n)
    save_vector!(io,spA.colptr)
    save_vector!(io,spA.rowval)
    save_vector!(io,spA.nzval)
end

function read_sparse_matrix!(io,T::DataType)
    m = read(io,Int64)
    n = read(io,Int64)
    colptr = read_vector!(io,Int)
    rowval = read_vector!(io,Int)
    nzval = read_vector!(io,T)
    @show m,n
    @show first(colptr),last(colptr)
    @show first(rowval),last(rowval)
    @show first(nzval),last(nzval)
    SparseMatrixCSC{Float64, Int64}(m,n,colptr,rowval,nzval)
end



function test_sparse(n)

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

    bc=(dirichlet,dirichlet,dirichlet,dirichlet)

    l2D=Laplacian2D(n,n,L,L,bc...)


    @time s2D=sparse_corr(l2D)

    io=open("toto.bin","w")
    xf=[Float64(i) for i ∈ 1:10]
    xi=[Int32(i) for i ∈ 1:10]

    save_vector!(io,xf)
    save_vector!(io,xi)
    close(io)

    io=open("toto.bin","r")
    xf=read_vector!(io,Float64)
    xi=read_vector!(io,Int32)
    close(io)

    @show xf,xi


    splu,tlu=PoissonSparseLU(s2D)

    @show issymmetric(s2D)

    min_time = Inf
    average_time = 0.0
    neval = 100

    for l ∈ 1:neval
        ts=@elapsed solve!(pref,b,splu)
        min_time = min(min_time,ts)
        average_time += ts
        # @show ts*1000
    end

    


    @show min_time*1000,"[ms]"
    @show (average_time/neval)*1000,"[ms]"

    @show first(pref),last(pref)
    @show first(b),last(b)

    io=open("sparse.bin","w")
    save_sparse_matrix!(io,s2D)     # A
    save_vector!(io,pref)           # x
    save_vector!(io,b)              # b
    close(io)

    io=open("sparse.bin","r")
    s2Dr=read_sparse_matrix!(io,Float64)
    close(io)

    @show s2D == s2Dr
    

    #convert coordinate to coordinate format

    Is,Js,Vs=findnz(s2D)
    newS = sparse(Is,Js,Vs)
    @show s2D == newS


    io=open("sparse_coordinate.bin","w")
    save_vector!(io,Is)     # Is
    save_vector!(io,Js)     # Js
    save_vector!(io,Vs)     # Vs
    save_vector!(io,pref)   # x
    save_vector!(io,b)      # b
    close(io)




    # xf=[Float64(i) for i ∈ 1:10]
    # # xi=[i for i ∈ 1:10]
    # for xi ∈ xf
    #     write(io,xi)
    # end
    # close(io)

    # io=open("toto.bin","r")
    # for i ∈ 1:10
    #     rxf=read(io,Float64)
    #     @show rxf
    # end
    # close(io)







    # @time s2D_nocorr=SparseArrays.sparse(l2D)
    # splu_nocorr,tlu_nocorr=PoissonSparseLU(s2D_nocorr)
    # stationary_methods=["jacobi","gauss_seidel","sor","ssor"]
    # for stm in stationary_methods
    #     test_stationary(msm,stm,s2D,splu,b)
    # end
    s2D
end

test_sparse(512)
