module LidJulMakieExt
using LidJul, Makie
function LidJul.plot_cavity(s::LidJul.CavityState)
    c=s.config
    x=((1:c.nx).-0.5).*(c.Lx/c.nx)
    y=((1:c.ny).-0.5).*(c.Ly/c.ny)
    uc=(s.u[1:end-1,:]+s.u[2:end,:])/2
    vc=(s.v[:,1:end-1]+s.v[:,2:end])/2
    fig=Figure(size=(1000,450))
    speed=Axis(fig[1,1],title="Speed at t=$(round(s.time,digits=3))",xlabel="x",ylabel="y",aspect=DataAspect())
    heatmap!(speed,x,y,hypot.(uc,vc))
    pressure=Axis(fig[1,2],title="Zero-mean pressure",xlabel="x",ylabel="y",aspect=DataAspect())
    heatmap!(pressure,x,y,s.p)
    fig
end
end
