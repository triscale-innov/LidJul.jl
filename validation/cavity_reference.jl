using LidJul, LinearAlgebra, TOML, Dates

function reference_data()
    rows=[split(line,',') for line in eachline(joinpath(@__DIR__,"ghia_re100.csv")) if !startswith(line,"#")]
    Dict(component=>[(coordinate=parse(Float64,row[2]),velocity=parse(Float64,row[3]))
                    for row in rows if row[1]==component] for component in ("u","v"))
end
function linear_sample(coordinates,values,point)
    i=clamp(searchsortedlast(coordinates,point),1,length(coordinates)-1)
    weight=(point-coordinates[i])/(coordinates[i+1]-coordinates[i])
    (1-weight)*values[i]+weight*values[i+1]
end
"""Compare the simulated centerlines to all published Re=100 samples."""
function reference_error(state)
    profiles=centerline_velocities(state);reference=reference_data()
    eu=[linear_sample(profiles.y,profiles.u,r.coordinate)-r.velocity for r in reference["u"]]
    ev=[linear_sample(profiles.x,profiles.v,r.coordinate)-r.velocity for r in reference["v"]]
    (u_rms=norm(eu)/sqrt(length(eu)),v_rms=norm(ev)/sqrt(length(ev)),
     rms=norm(vcat(eu,ev))/sqrt(length(eu)+length(ev)),max_error=max(maximum(abs,eu),maximum(abs,ev)))
end
"""
Run grid and timestep studies and optionally save a machine-readable TOML report.
Require actual steady convergence, accurate projection, bounded reference error
and decreasing differences between successive grids; these checks make failures visible instead of merely plotting profiles.
"""
function cavity_validation(;grids=(16,32,64),dt=.0025,tf=40,steady_tol=1e-7,output=nothing)
    rows=Dict{String,Any}[]
    profiles=Vector{Float64}[]
    for n in grids
        result=simulate_cavity(CavityConfig(;Re=100,nx=n,dt,tf,pressure_reltol=1e-11);steady_tol)
        errorset=reference_error(result.state)
        lines=centerline_velocities(result.state);reference=reference_data()
        samples=vcat([linear_sample(lines.y,lines.u,r.coordinate) for r in reference["u"]],
                     [linear_sample(lines.x,lines.v,r.coordinate) for r in reference["v"]])
        push!(profiles,samples)
        result.steady || Base.error("Re=100 simulation did not reach steady tolerance on grid $n")
        maximum(result.after)<1e-9 || Base.error("projection divergence is too large")
        wall_error(result.state)<1e-12 || Base.error("wall condition failed")
        push!(rows,Dict("grid"=>n,"centerline_samples"=>samples,"dt"=>Float64(result.state.dt),"time"=>Float64(result.state.time),
             "steps"=>result.state.steps,"u_rms"=>errorset.u_rms,"v_rms"=>errorset.v_rms,
             "rms"=>errorset.rms,"max_error"=>errorset.max_error,"max_divergence"=>maximum(result.after),
             "wall_error"=>wall_error(result.state),"velocity_change"=>result.state.velocity_change))
        println((grid=n,rms=errorset.rms,max_error=errorset.max_error,
                 time=result.state.time,divergence=maximum(result.after)))
    end
    errors=[row["rms"] for row in rows]
    last(errors)<.01 || error("fine-grid reference RMS error exceeds 0.01 lid velocities")
    spatial_differences=[norm(profiles[k+1]-profiles[k])/sqrt(length(profiles[k])) for k=1:length(profiles)-1]
    length(spatial_differences)<2 || last(spatial_differences)<.6first(spatial_differences) ||
        error("centerline grid refinement failed")
    states=[simulate_cavity(CavityConfig(;nx=16,dt=h,tf=.4)).state for h in (.02,.01,.005)]
    time_errors=[norm(states[k].u-states[3].u)+norm(states[k].v-states[3].v) for k=1:2]
    time_errors[2]<time_errors[1]/2 || error("timestep refinement failed")
    report=Dict("reference"=>"Ghia, Ghia and Shin (1982), DOI 10.1016/0021-9991(82)90058-4, Tables I and II, Re=100",
                "Re"=>100,"Lx"=>1,"Ly"=>1,"lid_velocity"=>1,"donor_cell"=>0,
                "steady_tol"=>steady_tol,"tf_max"=>tf,"pressure_reltol"=>1e-11,
                "spatial_differences"=>spatial_differences,"blas_threads"=>BLAS.get_num_threads(),
                "julia"=>string(VERSION),"generated_utc"=>string(now(UTC)),"grid_study"=>rows,
                "timestep_study"=>Dict("dt"=>[.02,.01,.005],"tf"=>.4,"grid"=>16,"errors_to_finest"=>time_errors))
    output===nothing || open(io->TOML.print(io,report),output,"w")
    report
end
