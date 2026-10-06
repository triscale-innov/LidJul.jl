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

"""Bilinearly interpolate the MAC velocity, with reflected tangential wall ghosts."""
function cavity_velocity(state, point)
    c = state.config
    x, y = point
    ux, uy = x * c.nx + 1, y * c.ny + 0.5
    vx, vy = x * c.nx + 0.5, y * c.ny + 1
    iu, ju = clamp(floor(Int, ux), 1, c.nx), clamp(floor(Int, uy), 0, c.ny)
    iv, jv = clamp(floor(Int, vx), 0, c.nx), clamp(floor(Int, vy), 1, c.ny)
    u(i, j) = j == 0 ? -state.u[i, 1] : j == c.ny + 1 ? 2c.lid_velocity - state.u[i, end] : state.u[i, j]
    v(i, j) = i == 0 ? -state.v[1, j] : i == c.nx + 1 ? -state.v[end, j] : state.v[i, j]
    blend(f, i, j, a, b) = (1-a) * ((1-b) * f(i, j) + b * f(i, j+1)) +
                          a * ((1-b) * f(i+1, j) + b * f(i+1, j+1))
    Point2f(blend(u, iu, ju, ux-iu, uy-ju), blend(v, iv, jv, vx-iv, vy-jv))
end

"""Record a computed Re=100 flow with smooth velocity streamlines and a fixed speed scale."""
function cavity_animation(; output=joinpath(MEDIA_DIRECTORY, "cavity_evolution.gif"),
                          nx=64, frames=120, steps_per_frame=25, dt=0.005, framerate=12)
    config = CavityConfig(; nx, Re=100, dt, tf=frames * steps_per_frame * dt, pressure_reltol=1e-11)
    state = CavityState(config)
    speed = Observable(zeros(nx, nx))
    velocity = Observable{Function}(point -> cavity_velocity(state, point))
    profile = Observable(centerline_velocities(state).u)
    status = Observable("t = 0.00 · RMS divergence = 0.0")
    cells = ((1:nx) .- 0.5) ./ nx
    profile_y = centerline_velocities(state).y
    background, foreground, accent = "#101a2b", "#edf3fc", "#6ee7d2"
    figure = Figure(size=(1080, 690), fontsize=18, backgroundcolor=background)
    Label(figure[0, 1:3], "LID-DRIVEN CAVITY", fontsize=28, font=:bold, color=foreground)
    Label(figure[1, 1:3], "Re = 100   ·   $(nx) × $(nx) MAC grid   ·   dt = $(dt)", fontsize=16, color=accent)
    function dark_axis(position; kwargs...)
        Axis(position; backgroundcolor=background, xlabelcolor=foreground, ylabelcolor=foreground,
             xticklabelcolor=foreground, yticklabelcolor=foreground, xtickcolor=foreground,
             ytickcolor=foreground, spinewidth=1, leftspinecolor="#63738b", bottomspinecolor="#63738b",
             rightspinevisible=false, topspinevisible=false, titlecolor=foreground,
             xgridvisible=false, ygridvisible=false, kwargs...)
    end
    flow = dark_axis(figure[2, 1]; xlabel="x / L", ylabel="y / L", aspect=DataAspect())
    heat = heatmap!(flow, cells, cells, speed; colorrange=(0, 1), colormap=:magma,
                    colorscale=sqrt, interpolate=true)
    streamplot!(flow, velocity, 0.004..0.996, 0.004..0.996;
                gridsize=(28, 28), density=0.6, stepsize=0.004, maxsteps=1400,
                color=_->RGBAf(0.91, 0.97, 1, 0.82), linewidth=1.15, arrow_size=7)
    lines!(flow, [Point2f(0.03, 1.025), Point2f(0.94, 1.025)]; color=accent, linewidth=3)
    scatter!(flow, [Point2f(0.94, 1.025)]; color=accent, marker=:rtriangle, markersize=15)
    xlims!(flow, 0, 1)
    ylims!(flow, 0, 1.06)
    Colorbar(figure[2, 2], heat; label="Speed / lid velocity", labelcolor=foreground,
             ticklabelcolor=foreground, tickcolor=foreground,
             leftspinecolor="#63738b", rightspinecolor="#63738b",
             topspinecolor="#63738b", bottomspinecolor="#63738b",
             ticks=[0, 0.1, 0.3, 0.6, 1], width=18)
    axis = dark_axis(figure[2, 3]; title="Vertical centerline", xlabel="u / lid velocity", ylabel="y / L")
    hlines!(axis, [0.5]; color=(:white, 0.12), linestyle=:dash)
    vlines!(axis, [0]; color=(:white, 0.12), linestyle=:dash)
    lines!(axis, profile, profile_y; color=accent, linewidth=3)
    xlims!(axis, -0.35, 1.05)
    ylims!(axis, 0, 1)
    colsize!(figure.layout, 1, Relative(0.62))
    Label(figure[3, 1:3], status; fontsize=16, color=foreground)
    Label(figure[4, 1:3], "Velocity streamlines with direction arrows · fixed speed scale (square-root color mapping)", fontsize=14, color="#acb9ce")
    record(figure, output, 1:(frames + 12); framerate, px_per_unit=1.25) do frame
        if frame <= frames
            for _ in 1:steps_per_frame
                step!(state)
            end
            uc = (state.u[1:end-1, :] + state.u[2:end, :]) / 2
            vc = (state.v[:, 1:end-1] + state.v[:, 2:end]) / 2
            speed[] = hypot.(uc, vc)
            notify(velocity)
            profile[] = centerline_velocities(state).u
            status[] = "t = $(round(state.time; digits=2))   ·   RMS divergence = $(round(state.divergence_after; sigdigits=2))"
        end
    end
    save(joinpath(dirname(output), "cavity_preview.png"), figure; px_per_unit=1.5)
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
