using LidJul

"""
Run a parameterized cavity. Set `visualize=true` after loading a Makie backend to
show the final solution. The default execution never creates a window.
"""
function cavity_example(;Re=100,dt=.005,tf=20,nx=32,ny=nx,visualize=false,kwargs...)
    result=simulate_cavity(CavityConfig(;Re,dt,tf,nx,ny,kwargs...))
    visualize && display(plot_cavity(result.state))
    result
end
