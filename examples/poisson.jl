using LidJul, LinearAlgebra, SparseArrays

"""Solve a manufactured mixed-boundary problem on a rectangular grid."""
function poisson_example(;nx=32,ny=64,T=Float64)
    lap=Laplacian2D(nx,ny,1,2,dirichlet,neumann,dirichlet,neumann;T)
    exact=T[sin(0.1i+0.2j) for i=1:nx,j=1:ny]
    rhs=reshape(sparse(lap)*vec(exact),nx,ny)
    outputs=Dict()
    for factory in (PoissonTTSolver,PoissonSparseLU,PoissonSparseAMG,PoissonGMG)
        x=zeros(T,nx,ny)
        result=solve!(x,rhs,factory(lap);reltol=T===Float32 ? 1e-4 : 1e-8)
        outputs[string(factory)]=(result=result,error=norm(x-exact)/norm(exact))
    end
    outputs
end
