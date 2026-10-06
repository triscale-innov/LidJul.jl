@testset "Interactive plots without a display server" begin
    a = Dict(1=>[1.0,2.0], 2=>[3.0], 3=>[4.0,5.0])
    b = Dict(1=>[0.0], 2=>[1.0], 3=>[2.0,3.0])
    fig = @test_logs (:warn,r"Vectors for pdt=1") plot_interactive(a,b)
    @test fig isa PlotlyBase.Plot
    @test length(fig.data) == 6
    @test fig.data[1][:visible] == true
    @test fig.data[4][:visible] == false
    @test fig.data[3][:y] == [2.0]
    steps = fig.layout[:sliders][1][:steps]
    @test [step[:label] for step in steps] == ["2","3"]
    @test steps[2][:args][1][:visible] == [false,false,false,true,true,true]
    io = IOBuffer()
    PlotlyBase.to_html(io,fig)
    @test occursin("Plotly.newPlot",String(take!(io)))
    @test_logs (:warn,r"No common keys") plot_interactive(Dict{Int,Vector{Float64}}(),b)
end
