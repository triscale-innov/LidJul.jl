using TOML, Printf

"""Render a README-sized comparison from recorded, accuracy-checked measurements."""
function render_summary(; report=joinpath(@__DIR__, "results.toml"),
                        output=joinpath(@__DIR__, "..", "docs", "src", "assets", "solver_comparison.md"))
    data = TOML.parsefile(report)
    rows = filter(data["cases"]) do row
        row["scalar_type"] == "Float64" && row["boundary"] == "DNDN" && row["reltol"] == 1e-8
    end
    methods = ("Tensor", "SparseLU", "GMG", "ILU_GMRES", "AMG", "CG")
    open(output, "w") do io
        println(io, "| Solver | Solve 16 × 32 (ms) | Solve 32 × 64 (ms) | Solve 64 × 64 (ms) | Setup 64 × 64 (ms) | Iterations at 64 × 64 |")
        println(io, "|:--|--:|--:|--:|--:|--:|")
        for method in methods
            selected = Dict((row["nx"], row["ny"]) => row for row in rows if row["solver"] == method)
            last = selected[(64, 64)]
            @assert all(row["converged"] && row["relative_residual"] <= 1e-8 for row in values(selected))
            times = [selected[grid]["solve_nanoseconds"] / 1e6 for grid in ((16, 32), (32, 64), (64, 64))]
            @printf(io, "| %s | %.3f | %.3f | %.3f | %.3f | %d |\n", replace(method, "_" => " + "),
                    times[1], times[2], times[3], last["setup_nanoseconds"] / 1e6, last["iterations"])
        end
    end
    readme = joinpath(@__DIR__, "..", "README.md")
    start_marker, end_marker = "<!-- benchmark-table:start -->", "<!-- benchmark-table:end -->"
    original = read(readme, String)
    @assert occursin(start_marker, original) && occursin(end_marker, original)
    updated = replace(original, Regex(start_marker * ".*?" * end_marker, "s") =>
                      start_marker * "\n\n" * read(output, String) * "\n" * end_marker)
    write(readme, updated)
    println("Updated README and wrote measured solver comparison to ", output)
    output
end

if abspath(PROGRAM_FILE) == @__FILE__
    render_summary()
end
