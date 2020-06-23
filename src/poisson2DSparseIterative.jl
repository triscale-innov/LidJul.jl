export PoissonSparseIterative,solve!
using IterativeSolvers

struct PoissonSparseIterative
    spLxy::SparseMatrixCSC{Float64,Int64}

    function PoissonSparseIterative(sp)
        new(sp)
    end
end

function optimal_w(x1D)
    h=1/sqrt(length(x1D))
    w=2/(1+sin(π*h))
end


solve_sor!(x,a,b,nstep) = IterativeSolvers.sor!(x,a,b,optimal_w(x),maxiter=nstep)
solve_ssor!(x,a,b,nstep) = IterativeSolvers.ssor!(x,a,b,optimal_w(x),maxiter=nstep)
solve_gauss_seidel!(x,a,b,nstep) = IterativeSolvers.gauss_seidel!(x,a,b,maxiter=nstep)
solve_jacobi!(x,a,b,nstep) = IterativeSolvers.jacobi!(x,a,b,maxiter=nstep)


function solve!(Xxy,Bxy,sp::PoissonSparseIterative,method_name::String="cg")
    b=view(Bxy,1:length(Bxy))
    x=view(Xxy,1:length(Xxy))
    A=sp.spLxy

    method_name=="cg" && return IterativeSolvers.cg!(x,A,b;log=true,tol = 1e-10,maxiter=1000)
    # method_name="jacobi" && solve_cg!(Xxy,Bxy,sp)
    # method_name="gauss_seidel" && solve_cg!(Xxy,Bxy,sp)



    # method_name=="gauss_seidel" && return IterativeSolvers.gauss_seidel!(x,A,b,maxiter=1000)
    method_name=="gauss_seidel" && return solve_stationary!(Xxy,Bxy,sp::PoissonSparseIterative,solve_gauss_seidel!)
    method_name=="jacobi" && return solve_stationary!(Xxy,Bxy,sp::PoissonSparseIterative,solve_jacobi!)
    method_name=="sor" && return solve_stationary!(Xxy,Bxy,sp::PoissonSparseIterative,solve_sor!)
    method_name=="ssor" && return solve_stationary!(Xxy,Bxy,sp::PoissonSparseIterative,solve_ssor!)



    # method_name="ssor" && solve_cg!(Xxy,Bxy,sp)

end




function solve_stationary!(Xxy,Bxy,sp::PoissonSparseIterative,iterative_algorithm)
    b=view(Bxy,1:length(Bxy))
    x=view(Xxy,1:length(Xxy))
    A=sp.spLxy

    #LP: stationary methods looks to have problem with views...
    x1D=zeros(length(Xxy))
    b1D=zeros(length(Xxy))
    copyto!(b1D,b)
    copyto!(x1D,x)


    residual=Vector{Float64}()
    nstep=10
    # h=1/size(Bxy,1)
    # w=2/(1+sin(π*h))
    # @show w
    res=typemax(Float64)
    iter=1
    while res>1.e-8 && iter<2000
        iterative_algorithm(x1D,A,b1D,nstep)
        res=norm(A*x1D-b1D)
        for i=1:nstep
            push!(residual,res)
        end
        iter+=nstep
    end
    @show iterative_algorithm,iter
    copyto!(x,x1D)
    x,residual
end
