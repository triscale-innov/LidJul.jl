export PoissonSparseAMG,solve!
using AlgebraicMultigrid
using IterativeSolvers

struct PoissonSparseAMG
    spLxy::SparseMatrixCSC{Float64,Int64}
    rgspLxy::Any

    function PoissonSparseAMG(sp)
        tmg=@elapsed rs=ruge_stuben(sp)
        println("AMG init time:",tmg)
        new(sp,rs)
    end
end

function solve!(Xxy,Bxy,sp::PoissonSparseAMG)
    ml = sp.rgspLxy
    p = aspreconditioner(ml)
    b=view(Bxy,1:length(Bxy))
    x=view(Xxy,1:length(Xxy))
    a,cgamglog=IterativeSolvers.cg!(x,sp.spLxy,b,Pl = p;log=true,tol = 1e-10,maxiter=1000)
    @show cgamglog
    cgamglog
end
