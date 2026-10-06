module LidJulPlotlyExt
using LidJul, PlotlyBase

function LidJul.plot_interactive(a::AbstractDict, b::AbstractDict)

    # Get common keys and sort them
    pdts = sort(collect(intersect(keys(a), keys(b))))

    filter!(pdts) do pdt
        valid = length(a[pdt]) == length(b[pdt])
        valid || @warn "Vectors for pdt=$pdt have different lengths. Skipping."
        valid
    end

    if isempty(pdts)
        @warn "No common keys with matching vector lengths to plot."
        return
    end

    # Create a figure with subplots
    fig = Plot(Layout(Subplots(rows=2, cols=1, shared_xaxes=true, vertical_spacing=0.1)))

    # Add traces for each pdt
    for pdt in pdts
        # Check that vectors have the same length
        if length(a[pdt]) != length(b[pdt])
            @warn "Vectors for pdt=$pdt have different lengths. Skipping."
            continue
        end

        is_visible = (pdt == pdts[1]) # Only the first pdt is visible initially

        # Trace for a on the top plot
        add_trace!(fig, scatter(
            x=1:length(a[pdt]),
            y=a[pdt],
            mode="lines",
            name="a (pdt=$pdt)",
            visible=is_visible
        ), row=1, col=1)

        # Trace for b on the top plot
        add_trace!(fig, scatter(
            x=1:length(b[pdt]),
            y=b[pdt],
            mode="lines",
            name="b (pdt=$pdt)",
            visible=is_visible
        ), row=1, col=1)

        # Trace for the difference on the bottom plot
        diff = a[pdt] - b[pdt]
        add_trace!(fig, scatter(
            x=1:length(diff),
            y=diff,
            mode="lines",
            name="a-b (pdt=$pdt)",
            line=attr(color="green"),
            visible=is_visible
        ), row=2, col=1)
    end

    # Create slider steps
    steps = []
    for (i, pdt) in enumerate(pdts)
        # Each pdt corresponds to 3 traces
        visibility = [j in (3*i-2):(3*i) for j in 1:length(fig.data)]

        step = attr(
            label = string(pdt),
            method = "update",
            args = [attr(visible = visibility),
                    attr(title = "Showing data for pdt = $pdt")]
        )
        push!(steps, step)
    end

    # Relayout the figure with a slider and titles
    relayout!(fig,
        title_text="Showing data for pdt = $(pdts[1])",
        yaxis_title="Value",
        yaxis2_title="Difference (a-b)",
        xaxis2_title="Index",
        sliders=[attr(
            active=0,
            currentvalue=attr(prefix="pdt: "),
            pad=attr(t=50),
            steps=steps
        )]
    )

    return fig
end


end
