# This script tests the interactive plotting function.
using PlotlyJS

# We include the source file directly to avoid loading the entire LidJul module,
# which has a dependency on GLMakie that fails in headless environments.
include("../src/interactive_plotting.jl")

# 1. Create sample data
# Using a larger vector size to make the lines more visible.
a = Dict(i => rand(20) .* i for i in 10:5:100)
b = Dict(i => rand(20) .* i .+ 5 for i in 10:5:100)

# 2. Call the plotting function
# This should generate a plot object from PlotlyJS
p = plot_interactive(a, b)

# 3. Save the plot to a file
# This will allow us to verify that the plot is generated without having to display it.
output_filename = "interactive_plot_with_slider.html"
savefig(p, output_filename)

println("Interactive plot saved to $(output_filename)")
