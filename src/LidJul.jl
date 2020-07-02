module LidJul

    export BoundaryCondition,dirichlet,neumann

    @enum BoundaryCondition dirichlet=0 neumann=1

    include("laplacian1D.jl")
    include("laplacian2D.jl")
    # include("poisson2D_ADI.jl")
    # include("poisson2D_CG.jl")
    # include("poisson2D_PCG.jl")
    include("poisson2D_TT.jl")
    # include("poisson2DSparse.jl")
    include("poisson2DSparseLU.jl")
    include("poisson2DSparseAMG.jl")
    include("poisson2DSparseCGILU.jl")
    include("poisson2DSparseIterative.jl")
    # include("ADISmoother.jl")
    include("GSSmoother.jl")
    # include("GSRBSmoother.jl")
    # include("CGSmoother.jl")
    # include("BlockJacobiSmoother.jl")

    include("poisson2DGMG.jl")
    include("poisson2DGMG_new.jl")
    # include("RedBlackArray.jl")
    # include("poisson2DGMGRB.jl")
    include("TensorialOperator.jl")



    # include("poisson2DSparse.jl")
end # module
