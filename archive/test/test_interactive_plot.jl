# This script tests the interactive plotting function.
using LidJul, PlotlyBase

# 1. Create sample data
# Using a larger vector size to make the lines more visible.
a = Dict(i => rand(20) .* i for i in 10:5:100)
b = Dict(i => rand(20) .* i .+ 5 for i in 10:5:100)

# 2. Call the plotting function
# This should generate a plot object from PlotlyBase
p = plot_interactive(a, b)

# 3. Save the plot to a file
# This will allow us to verify that the plot is generated without having to display it.
output_filename = "interactive_subplots.html"
open(output_filename, "w") do io
    PlotlyBase.to_html(io, p)
end

println("Interactive plot saved to $(output_filename)")
