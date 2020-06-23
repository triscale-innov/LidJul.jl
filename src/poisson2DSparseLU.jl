export PoissonSparseLU,solve!,mul!
using SparseArrays
using LinearAlgebra

struct PoissonSparseLU
    spLxy::SparseMatrixCSC{Float64,Int64}
    fspLxy::Any

    function PoissonSparseLU(sp)
        tlu=@elapsed fsp=lu(sp)
        println("LU factorization time: ",tlu)
        new(sp,fsp),tlu
    end
end

function solve!(Xxy,Bxy,sp::PoissonSparseLU)
    ldiv!(view(Xxy,1:length(Xxy)),sp.fspLxy,view(Bxy,1:length(Bxy)))
    Xxy
end

function mul!(Bxy,sp::PoissonSparseLU,Xxy)
    LinearAlgebra.mul!(view(Bxy,1:length(Xxy)),sp.spLxy,view(Xxy,1:length(Xxy)))
end
