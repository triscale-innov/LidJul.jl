using LidJul, CairoMakie, TOML

"""Plot the recorded offline Ghia comparison; run the validation study first."""
function validation_plot(;report=joinpath(@__DIR__,"..","validation","results.toml"))
    data=TOML.parsefile(report)
    rows=[split(line,',') for line in eachline(joinpath(@__DIR__,"..","validation","ghia_re100.csv")) if !startswith(line,"#")]
    coordinates=Dict(c=>[parse(Float64,r[2]) for r in rows if r[1]==c] for c in ("u","v"))
    values=Dict(c=>[parse(Float64,r[3]) for r in rows if r[1]==c] for c in ("u","v"))
    fig=Figure(size=(1100,500))
    au=Axis(fig[1,1],title="Vertical centerline, Re=100",xlabel="u / lid velocity",ylabel="y / L")
    av=Axis(fig[1,2],title="Horizontal centerline, Re=100",xlabel="x / L",ylabel="v / lid velocity")
    for grid in data["grid_study"]
        samples=grid["centerline_samples"];n=length(coordinates["u"])
        lines!(au,samples[1:n],coordinates["u"],label="$(grid["grid"]) × $(grid["grid"]) cells")
        lines!(av,coordinates["v"],samples[n+1:end],label="$(grid["grid"]) × $(grid["grid"]) cells")
    end
    scatter!(au,values["u"],coordinates["u"],label="Ghia et al. (1982)",color=:black,markersize=7)
    scatter!(av,coordinates["v"],values["v"],label="Ghia et al. (1982)",color=:black,markersize=7)
    Legend(fig[2,1:2],au;orientation=:horizontal)
    fig
end
