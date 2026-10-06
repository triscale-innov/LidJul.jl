module LidJul

using LinearAlgebra, SparseArrays
import IterativeSolvers
import AlgebraicMultigrid
import IncompleteLU

export BoundaryCondition, dirichlet, neumann, Laplacian1D, Laplacian2D, sparse_corr
export AbstractPoissonSolver, SolveResult, solve!, residual_norm
export PoissonSparseLU, PoissonSparseAMG, PoissonSparseCGILU, PoissonSparseIterative
export TensorialOperator, PoissonTTSolver, PoissonGMG, PoissonGMG_new, GSSmoother, maxlevels
export CavityConfig, CavityState, CavityResult, simulate_cavity, step!, divergence_norm
export centerline_velocities, wall_error, plot_interactive, plot_cavity

"""Homogeneous boundary condition: [`dirichlet`](@ref) or [`neumann`](@ref)."""
@enum BoundaryCondition dirichlet=0 neumann=1
@doc "Homogeneous Dirichlet condition: the value at the wall is zero." dirichlet
@doc "Homogeneous Neumann condition: the normal derivative at the wall is zero." neumann

include("laplacian1D.jl")
include("laplacian2D.jl")
include("solver_interface.jl")
include("TensorialOperator.jl")
include("poisson2D_TT.jl")
include("poisson2DSparseLU.jl")
include("poisson2DSparseAMG.jl")
include("poisson2DSparseCGILU.jl")
include("poisson2DSparseIterative.jl")
include("GSSmoother.jl")
include("poisson2DGMG.jl")
include("cavity.jl")

"""
    plot_interactive(a, b)

Create a Plotly comparison with a synchronized slider. Load `PlotlyBase` first
to enable this optional extension. Returns a `PlotlyBase.Plot`; mismatched
vector lengths are skipped. The numerical package has no plotting dependency.
"""
function plot_interactive end

"""
    plot_cavity(state)

Create a velocity/pressure figure for a [`CavityState`](@ref). This is an optional
interface enabled by loading a Makie backend (`CairoMakie` or `GLMakie`).
The numerical simulation never opens a window.
"""
function plot_cavity end

end
