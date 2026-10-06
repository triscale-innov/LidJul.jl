```@meta
CurrentModule = LidJul
DocTestSetup = :(using LidJul)
```

# Cavity simulation

```julia
using LidJul
config = CavityConfig(Re=100, nx=32, ny=48, dt=0.005, tf=20)
result = simulate_cavity(config)
profiles = centerline_velocities(result.state)
println((divergence=maximum(result.after), wall_error=wall_error(result.state)))
```

The default run has no graphics dependency and creates no windows. Change
Reynolds number, domain dimensions, grid dimensions, lid velocity, timestep,
final time and pressure solver through `CavityConfig`. The effective timestep
is reduced slightly when necessary to reach the final time exactly.

## Numerical method

Horizontal velocities live on vertical faces, vertical velocities on horizontal
faces, and pressure at cell centers. Conservative central fluxes advance
convection explicitly. `donor_cell` adds controllable first-order upwind damping.
The default is zero to preserve centered spatial accuracy. Viscous diffusion uses
implicit Euler and cached separable Helmholtz solves. Reflected tangential ghost
values impose no slip; the top lid has its prescribed velocity. The upper-lid
forcing uses the vertical grid spacing, including on rectangular grids.

The previous pressure gradient is included in the predictor. A homogeneous
Neumann pressure-increment solve projects the intermediate velocity onto the
discrete divergence-free space, then updates the pressure. This incremental
projection avoids a fixed-timestep steady-state error floor at the walls. Pressure uses a zero-mean gauge. Each step checks
the advective CFL and pressure convergence and records divergence before/after
the projection. The time discretization is first order; splitting near walls can
also influence spatial error. A small divergence alone does not establish
accuracy of the velocity solution: use the refinement studies described under
[Validation](@ref).

`steady_tol` is an optional stopping threshold for the RMS velocity time
derivative. With its default value zero, the run reaches `tf`. Results record
whether the final diagnostic meets the threshold. A stationary lid produces an
exactly stationary solution from the default initial condition.

## Plotting

Install the examples environment and load a Makie backend to enable the extension:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
```

```julia
using LidJul, CairoMakie
result = simulate_cavity(CavityConfig(nx=32, tf=2))
save("cavity.png", plot_cavity(result.state))
```

`examples/cavity.jl` exposes `cavity_example(; visualize=false, ...)`;
`examples/visualization.jl` provides a modern multigrid residual plot.
`callback(state)` can update a display or collect selected outputs during a run.
Plotly comparison sliders are provided by `plot_interactive` after loading
`PlotlyBase`. Loading LidJul alone does not load either plotting package.

`CairoMakie` exports figures without OpenGL; use the separate `examples/interactive/` environment for `GLMakie` windows on a
compatible desktop. Both activate the same optional Makie extension. Separating
the interactive environment keeps OpenGL precompilation out of the default
examples installation.

```sh
julia --project=examples/interactive -e 'using Pkg; Pkg.instantiate()'
```

```julia
using LidJul, GLMakie
isempty(GLMakie.GLFW.GetMonitors()) && error("No graphical monitor is available; use CairoMakie.")
result = simulate_cavity(CavityConfig(nx=32, tf=2))
display(plot_cavity(result.state))
```

Run the guarded desktop example with
`julia --project=examples/interactive examples/interactive/cavity.jl`.
The interactive project disables GLMakie's optional precompile workload using
`[preferences.GLMakie] precompile_workload = false`: that workload renders
OpenGL scenes even before a user calls `display`. This avoids requiring a
display during precompilation, at the cost of extra compilation on the first
plot. It does not provide a headless OpenGL renderer.

In the local sandboxed macOS session, GLFW reports zero monitors and a null
primary monitor. Without that preference, GLMakie 0.13.15 crashes with signal 11
in `_glfwGetMonitorPosCocoa` while precompiling. With the preference, importing
GLMakie succeeds; desktop rendering remains unverified because no monitor is
available to that process. See Makie's [headless guide](https://docs.makie.org/stable/explanations/headless.html)
for the graphics requirements on other systems.
