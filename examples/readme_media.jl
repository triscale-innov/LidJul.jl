using LidJul, CairoMakie, LinearAlgebra, TOML

const MEDIA_DIRECTORY = joinpath(@__DIR__, "..", "docs", "src", "assets")

"""Export the measured solver comparison from the checked-in benchmark report."""
function benchmark_figure(; output=joinpath(MEDIA_DIRECTORY, "solver_timings.svg"))
    report = TOML.parsefile(joinpath(@__DIR__, "..", "benchmark", "results.toml"))
    rows = filter(report["cases"]) do row
        row["scalar_type"] == "Float64" && row["boundary"] == "DNDN" && row["reltol"] == 1e-8
    end
    methods = ["Tensor", "SparseLU", "GMG", "ILU_GMRES", "AMG", "CG"]
    figure = Figure(size=(1050, 520), fontsize=17)
    Label(figure[0, 1], "Poisson solver timings", fontsize=25, font=:bold)
    Label(figure[1, 1], "Float64 · mixed boundaries · relative tolerance 10⁻⁸ · Apple M1 · one BLAS thread", fontsize=15)
    axis = Axis(figure[2, 1], xlabel="Pressure cells", ylabel="Cached solve time (ms)",
                xscale=log2, yscale=log10, xticks=([512, 2048, 4096], ["16 × 32", "32 × 64", "64 × 64"]))
    for method in methods
        selected = sort(filter(row -> row["solver"] == method, rows); by=row -> row["nx"] * row["ny"])
        scatterlines!(axis, [row["nx"] * row["ny"] for row in selected],
                      [row["solve_nanoseconds"] / 1e6 for row in selected]; label=replace(method, "_" => " + "), markersize=10, linewidth=2.5)
    end
    Legend(figure[3, 1], axis; orientation=:horizontal, nbanks=2)
    Label(figure[4, 1], "Minimum warmed trials; setup excluded. These are local measurements, not a universal ranking.", fontsize=14)
    save(output, figure)
    figure
end

"""Record a real Re=100 transient using CairoMakie, without an OpenGL display."""
function cavity_animation(; output=joinpath(MEDIA_DIRECTORY, "cavity_evolution.gif"),
                          nx=32, frames=120, steps_per_frame=25, dt=0.005, framerate=12)
    config = CavityConfig(; nx, Re=100, dt, tf=frames * steps_per_frame * dt, pressure_reltol=1e-11)
    state = CavityState(config)
    speed = Observable(zeros(nx, nx))
    streamfunction = Observable(zeros(nx + 1, nx + 1))
    profile = Observable(centerline_velocities(state).u)
    status = Observable("t = 0.00 · divergence = 0.0")
    cells = ((1:nx) .- 0.5) ./ nx
    nodes = range(0, 1; length=nx + 1)
    profile_y = centerline_velocities(state).y
    figure = Figure(size=(960, 500), fontsize=17)
    Label(figure[0, 1:3], "Lid-driven cavity · Re = 100", fontsize=25, font=:bold)
    Label(figure[1, 1:3], "$(nx) × $(nx) staggered grid · incremental pressure projection · dt = $(dt)", fontsize=15)
    flow = Axis(figure[2, 1], title="Speed and streamlines", xlabel="x / L", ylabel="y / L", aspect=DataAspect())
    heat = heatmap!(flow, cells, cells, speed; colorrange=(0, 1), colormap=:viridis)
    contour!(flow, nodes, nodes, streamfunction; levels=collect(-0.10:0.01:-0.01), color=(:white, 0.8), linewidth=1.5)
    Colorbar(figure[2, 2], heat; label="Speed / lid velocity")
    axis = Axis(figure[2, 3], title="Vertical centerline", xlabel="u / lid velocity", ylabel="y / L")
    lines!(axis, profile, profile_y; color=:steelblue, linewidth=3)
    xlims!(axis, -0.35, 1.05)
    ylims!(axis, 0, 1)
    Label(figure[3, 1:3], status; fontsize=15)
    record(figure, output, 1:(frames + 12); framerate) do frame
        if frame <= frames
            for _ in 1:steps_per_frame
                step!(state)
            end
            uc = (state.u[1:end-1, :] + state.u[2:end, :]) / 2
            vc = (state.v[:, 1:end-1] + state.v[:, 2:end]) / 2
            speed[] = hypot.(uc, vc)
            psi = zeros(nx + 1, nx + 1)
            psi[:, 2:end] = cumsum(state.u; dims=2) / nx
            streamfunction[] = psi
            profile[] = centerline_velocities(state).u
            status[] = "t = $(round(state.time; digits=2)) · RMS divergence = $(round(state.divergence_after; sigdigits=2))"
        end
    end
    save(joinpath(dirname(output), "cavity_preview.png"), figure)
    println("Cavity animation: t=", state.time, ", divergence=", state.divergence_after, ", wall error=", wall_error(state))
    @assert state.divergence_after < 1e-10 && wall_error(state) == 0
    state
end

"""Animate actual GMG V-cycles and their physical residuals on a rectangular grid."""
function multigrid_animation(; output=joinpath(MEDIA_DIRECTORY, "multigrid_convergence.gif"))
    lap = Laplacian2D(32, 64, 1, 1.5, dirichlet, neumann, dirichlet, neumann)
    rhs = ones(32, 64)
    solution = zeros(32, 64)
    solver = PoissonGMG(lap)
    exact = zeros(32, 64)
    solve!(exact, rhs, PoissonTTSolver(lap); reltol=1e-10)
    field = Observable(copy(solution))
    history = Observable(Point2f[(0, norm(rhs))])
    status = Observable("Initial iterate · physical residual = $(round(norm(rhs); digits=2))")
    figure = Figure(size=(960, 480), fontsize=17)
    Label(figure[0, 1:3], "Geometric multigrid · one V-cycle per frame", fontsize=24, font=:bold)
    flow = Axis(figure[1, 1], title="Poisson solution · 32 × 64", xlabel="x / Lx", ylabel="y / Ly")
    heat = heatmap!(flow, ((1:32) .- 0.5) / 32, ((1:64) .- 0.5) / 64, field;
                    colorrange=(0, maximum(exact)), colormap=:viridis)
    Colorbar(figure[1, 2], heat; label="Solution")
    axis = Axis(figure[1, 3], xlabel="V-cycle", ylabel="Physical residual norm", yscale=log10)
    scatterlines!(axis, history; color=:steelblue, linewidth=3, markersize=9)
    xlims!(axis, 0, 12)
    ylims!(axis, 1e-9, 100)
    hlines!(axis, [1e-9 * norm(rhs)]; color=:darkorange, linestyle=:dash)
    Label(figure[2, 1:3], status; fontsize=15)
    cycle = 0
    converged = false
    record(figure, output, 0:20; framerate=4) do frame
        if frame > 0 && !converged
            result = solve!(solution, rhs, solver; reltol=1e-9, maxiter=1, store_history=false)
            cycle += result.iterations
            converged = result.converged
            field[] = copy(solution)
            history[] = vcat(history[], Point2f[(cycle, result.residual_norm)])
            status[] = "V-cycle $(cycle) · physical residual = $(round(result.residual_norm; sigdigits=3))$(converged ? " · converged" : "")"
        end
    end
    save(joinpath(dirname(output), "multigrid_preview.png"), figure)
    @assert converged
    println("Multigrid animation: converged after ", cycle, " V-cycles")
    figure
end

if abspath(PROGRAM_FILE) == @__FILE__
    mkpath(MEDIA_DIRECTORY)
    benchmark_figure()
    cavity_animation()
    multigrid_animation()
end
