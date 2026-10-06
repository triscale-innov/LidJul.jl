# Run directly with `julia --startup-file=no --project=. test/cli_smoke.jl`.
include(joinpath(@__DIR__, "..", "examples", "poisson.jl"))
include(joinpath(@__DIR__, "..", "examples", "cavity.jl"))

for T in (Float32, Float64)
    outputs = poisson_example(; nx=16, ny=32, T)
    @assert all(value.result.converged for value in values(outputs))
    println("Poisson ", T, ": all four solvers converged")
end
result = cavity_example(; nx=16, ny=32, tf=0.02)
@assert divergence_norm(result.state) < 1e-10
@assert wall_error(result.state) == 0
println("Cavity: divergence = ", divergence_norm(result.state), ", wall error = 0")
