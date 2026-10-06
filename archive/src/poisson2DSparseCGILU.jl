export PoissonSparseCGILU,solve!
using IterativeSolvers
using IncompleteLU

struct PoissonSparseCGILU
    spLxy::SparseMatrixCSC{Float64,Int64}
    iluLxy::Any

    function PoissonSparseCGILU(sp)
        tilu=@elapsed lu = ilu(sp, τ = 0.2)
        println("ILU factorization time:",tilu)
        new(sp,lu)
    end
end

function solve!(Xxy,Bxy,sp::PoissonSparseCGILU)
    p = sp.iluLxy
    b=view(Bxy,1:length(Bxy))
    x=view(Xxy,1:length(Xxy))
    a,cgilulog=IterativeSolvers.cg!(x,sp.spLxy,b,Pl = p;log=true,reltol = 1e-16,maxiter=1000)
    @show cgilulog
    cgilulog
end
