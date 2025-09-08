# This file will contain the interactive plotting function.
using PlotlyJS

"""
    plot_interactive(a, b)

Creates an interactive plot of the values in dictionaries `a` and `b` using
PlotlyJS. The plot has two lines, one for `a` and one for `b`, corresponding
to the data for a given `pdt`. A slider allows the user to select which `pdt`
to display.

# Arguments
- `a::Dict{Int, Vector{Float64}}`: A dictionary with integer keys (`pdts`) and vectors of float values.
- `b::Dict{Int, Vector{Float64}}`: A dictionary with integer keys (`pdts`) and vectors of float values.
"""
function plot_interactive(a::Dict{Int, Vector{Float64}}, b::Dict{Int, Vector{Float64}})

    # Get common keys and sort them
    pdts = sort(collect(intersect(keys(a), keys(b))))

    if isempty(pdts)
        @warn "No common keys to plot."
        return
    end

    # Create all traces, two for each pdt
    traces = GenericTrace[]
    for pdt in pdts
        # Trace for a
        push!(traces, scatter(
            x=1:length(a[pdt]),
            y=a[pdt],
            mode="lines",
            name="a (pdt=$pdt)",
            visible=(pdt == pdts[1]) # Only the first pdt is visible initially
        ))
        # Trace for b
        push!(traces, scatter(
            x=1:length(b[pdt]),
            y=b[pdt],
            mode="lines",
            name="b (pdt=$pdt)",
            visible=(pdt == pdts[1])
        ))
    end

    # Create slider steps
    steps = []
    for (i, pdt) in enumerate(pdts)
        # Create a boolean array for visibility.
        # The i-th pair of traces should be visible.
        visibility = [j == 2*i-1 || j == 2*i for j in 1:length(traces)]

        step = attr(
            label = string(pdt),
            method = "update",
            args = [attr(visible = visibility),
                    attr(title = "Showing data for pdt = $pdt")]
        )
        push!(steps, step)
    end

    # Define the layout with a slider
    layout = Layout(
        title="Showing data for pdt = $(pdts[1])",
        xaxis=attr(title="Index"),
        yaxis=attr(title="Value"),
        sliders=[attr(
            active=0,
            currentvalue=attr(prefix="pdt: "),
            pad=attr(t=50),
            steps=steps
        )]
    )

    # Create the plot
    plot(traces, layout)
end

# Example Usage
function run_example()
    # Sample data
    a = Dict(i => rand(20) .* i for i in 10:5:50)
    b = Dict(i => rand(20) .* i .+ 5 for i in 10:5:50)

    # Generate the plot
    p = plot_interactive(a, b)

    return p
end
