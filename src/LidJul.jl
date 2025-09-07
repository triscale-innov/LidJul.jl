module LidJul

    export BoundaryCondition,dirichlet,neumann

    @enum BoundaryCondition dirichlet=0 neumann=1

    include("laplacian1D.jl")
    include("laplacian2D.jl")
    include("poisson2D_TT.jl")
    include("poisson2DSparseLU.jl")
    include("poisson2DSparseAMG.jl")
    include("poisson2DSparseCGILU.jl")
    include("poisson2DSparseIterative.jl")
    include("GSSmoother.jl")
    include("poisson2DGMG.jl")
    include("TensorialOperator.jl")

end # module
