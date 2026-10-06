using LidJul, CairoMakie

"""Plot the physical residual history of the unified geometric multigrid solver."""
function multigrid_example(;nx=32,ny=64)
    lap=Laplacian2D(nx,ny,1,2,dirichlet,neumann,dirichlet,neumann)
    x=zeros(nx,ny);b=ones(nx,ny)
    result=solve!(x,b,PoissonGMG(lap);reltol=1e-9)
    figure=Figure(size=(900,400))
    ax=Axis(figure[1,1],xlabel="V-cycle",ylabel="Physical residual norm",yscale=log10)
    lines!(ax,0:length(result.history)-1,result.history)
    solution=Axis(figure[1,2],title="Poisson solution",xlabel="Cell index x",ylabel="Cell index y")
    heatmap!(solution,x)
    figure
end
