"""
    CavityConfig(; Re=100, dt=0.01, tf=20, nx=32, ny=nx, Lx=1, Ly=1,
                   lid_velocity=1, donor_cell=0, pressure_solver=:tensor,
                   pressure_reltol=sqrt(eps(T)), pressure_maxiter=1000, T=Float64)

Parameters for an incompressible, two-dimensional lid-driven cavity on a
staggered MAC grid. `Re` defines viscosity as `abs(lid_velocity)*Lx/Re` (unit
reference velocity when the lid is stationary). Convection uses conservative
central differences with optional donor-cell damping in `[0,1]`; diffusion is
implicit Euler. Pressure is advanced by an incremental velocity projection. `dt` is an upper
bound: an integer number of equal steps reaches `tf` exactly. The advective CFL
limit is checked each step. Pressure solvers are `:tensor`, `:gmg`, `:lu`, `:amg`.
"""
struct CavityConfig{T<:AbstractFloat}
    Re::T
    dt::T
    tf::T
    nx::Int
    ny::Int
    Lx::T
    Ly::T
    lid_velocity::T
    donor_cell::T
    pressure_solver::Symbol
    pressure_reltol::T
    pressure_maxiter::Int
end
function CavityConfig(;Re=100,dt=0.01,tf=20,nx=32,ny=nx,Lx=1,Ly=1,
                      lid_velocity=1,donor_cell=0,pressure_solver=:tensor,
                      T::Type{<:AbstractFloat}=Float64,pressure_reltol=sqrt(eps(T)),pressure_maxiter=1000)
    T in (Float32,Float64) || throw(ArgumentError("supported types are Float32 and Float64"))
    all(x->isfinite(x)&&x>0,(Re,dt,tf,Lx,Ly)) || throw(ArgumentError("Re, dt, tf and lengths must be finite and positive"))
    nx isa Integer && ny isa Integer && min(nx,ny)>=4 || throw(ArgumentError("grid dimensions must be integers at least four"))
    isfinite(lid_velocity) || throw(ArgumentError("lid velocity must be finite"))
    isfinite(donor_cell) && 0<=donor_cell<=1 || throw(ArgumentError("donor_cell must be in [0,1]"))
    pressure_solver in (:tensor,:gmg,:lu,:amg) || throw(ArgumentError("unknown pressure solver"))
    isfinite(pressure_reltol) && pressure_reltol>0 || throw(ArgumentError("pressure tolerance must be finite and positive"))
    pressure_maxiter isa Integer && pressure_maxiter>0 || throw(ArgumentError("pressure_maxiter must be a positive integer"))
    pressure_solver===:gmg && maxlevels(nx,ny)
    CavityConfig{T}(T(Re),T(dt),T(tf),nx,ny,T(Lx),T(Ly),T(lid_velocity),T(donor_cell),
                    pressure_solver,T(pressure_reltol),pressure_maxiter)
end

"""
    CavityState(config)

Mutable cavity state with cached pressure and diffusion solvers. `u` has size
`(nx+1,ny)`, `v` has size `(nx,ny+1)`, and cell-centered `p` has size `(nx,ny)`.
Normal wall velocities are stored explicitly; tangential wall values are imposed
by reflected ghost values. Each state owns its workspaces and can run independently.
"""
mutable struct CavityState{T<:AbstractFloat,P,U,V}
    config::CavityConfig{T}
    u::Matrix{T}
    v::Matrix{T}
    p::Matrix{T}
    pressure_increment::Matrix{T}
    rhsu::Matrix{T}
    rhsv::Matrix{T}
    rhsp::Matrix{T}
    previous_u::Matrix{T}
    previous_v::Matrix{T}
    pressure::P
    diffusion_u::U
    diffusion_v::V
    dt::T
    time::T
    steps::Int
    divergence_before::T
    divergence_after::T
    velocity_change::T
end
function CavityState(c::CavityConfig{T}) where T
    dt=c.tf/ceil(Int,c.tf/c.dt)
    hx,hy=c.Lx/c.nx,c.Ly/c.ny
    viscosity=(c.lid_velocity==0 ? one(T) : abs(c.lid_velocity))*c.Lx/c.Re
    diffusion=dt*viscosity
    su=TensorialOperator(Array,T,c.nx-1,c.ny,hx,hy,2,3,diffusion,diffusion,1)
    sv=TensorialOperator(Array,T,c.nx,c.ny-1,hx,hy,3,2,diffusion,diffusion,1)
    lap=Laplacian2D(c.nx,c.ny,c.Lx,c.Ly,neumann,neumann,neumann,neumann;T)
    factory=c.pressure_solver===:tensor ? PoissonTTSolver : c.pressure_solver===:gmg ? PoissonGMG :
            c.pressure_solver===:lu ? PoissonSparseLU : PoissonSparseAMG
    pressure=factory(lap)
    CavityState(c,zeros(T,c.nx+1,c.ny),zeros(T,c.nx,c.ny+1),zeros(T,c.nx,c.ny),zeros(T,c.nx,c.ny),
                zeros(T,c.nx-1,c.ny),zeros(T,c.nx,c.ny-1),zeros(T,c.nx,c.ny),
                zeros(T,c.nx+1,c.ny),zeros(T,c.nx,c.ny+1),pressure,su,sv,dt,zero(T),0,zero(T),zero(T),zero(T))
end

@inline function _u(s,i,j)
    j==0 && return -s.u[i,1]
    j==s.config.ny+1 && return 2s.config.lid_velocity-s.u[i,end]
    s.u[i,j]
end
@inline function _v(s,i,j)
    i==0 && return -s.v[1,j]
    i==s.config.nx+1 && return -s.v[end,j]
    s.v[i,j]
end
@inline _flux(a,b,velocity,gamma)=(a+b)*velocity/2-gamma*abs(velocity)*(b-a)/2
function _divergence!(out,s)
    c=s.config; hx,hy=c.Lx/c.nx,c.Ly/c.ny
    @inbounds for j=1:c.ny,i=1:c.nx
        out[i,j]=(s.u[i+1,j]-s.u[i,j])/hx+(s.v[i,j+1]-s.v[i,j])/hy
    end
    out
end

"""
    divergence_norm(state)

Root-mean-square discrete divergence over all pressure cells. A successful
projection reduces this to the pressure solve accuracy divided by the cell count.
"""
divergence_norm(s::CavityState)=norm(_divergence!(s.rhsp,s))/sqrt(length(s.rhsp))

"""
    wall_error(state)

Maximum wall-velocity discrepancy. Normal velocities are measured from stored
boundary faces; tangential no-slip values are reconstructed from ghost averages.
The moving top lid has its prescribed velocity, with corner values excluded.
"""
function wall_error(s::CavityState{T}) where T
    c=s.config
    error=max(maximum(abs,view(s.u,1,:)),maximum(abs,view(s.u,c.nx+1,:)),
              maximum(abs,view(s.v,:,1)),maximum(abs,view(s.v,:,c.ny+1)))
    for i=2:c.nx
        error=max(error,abs((_u(s,i,0)+_u(s,i,1))/2),
                  abs((_u(s,i,c.ny)+_u(s,i,c.ny+1))/2-c.lid_velocity))
    end
    for j=2:c.ny
        error=max(error,abs((_v(s,0,j)+_v(s,1,j))/2),
                  abs((_v(s,c.nx,j)+_v(s,c.nx+1,j))/2))
    end
    T(error)
end

"""
    step!(state)

Advance one cached timestep with explicit conservative convection, implicit
viscous diffusion and a pressure projection. Return the pressure [`SolveResult`](@ref).
Reject an unstable advective CFL or an unconverged pressure solve. Diagnostics
`divergence_before`, `divergence_after` and `velocity_change` are updated in place.
"""
function step!(s::CavityState{T}) where T
    c=s.config; nx,ny=c.nx,c.ny
    s.steps<ceil(Int,c.tf/c.dt) || throw(ArgumentError("the configured final time has been reached"))
    hx,hy=c.Lx/nx,c.Ly/ny
    cfl=s.dt*(max(maximum(abs,s.u),abs(c.lid_velocity))/hx+maximum(abs,s.v)/hy)
    cfl<=1 || throw(ArgumentError("advective CFL exceeds one; reduce dt"))
    gamma=c.donor_cell
    @inbounds for j=1:ny,i=2:nx
        uc=_u(s,i,j)
        ue,uw=_u(s,i+1,j),_u(s,i-1,j)
        un,us=_u(s,i,j+1),_u(s,i,j-1)
        vn=(_v(s,i-1,j+1)+_v(s,i,j+1))/2
        vs=(_v(s,i-1,j)+_v(s,i,j))/2
        adv=(_flux(uc,ue,(uc+ue)/2,gamma)-_flux(uw,uc,(uw+uc)/2,gamma))/hx+
            (_flux(uc,un,vn,gamma)-_flux(us,uc,vs,gamma))/hy
        s.rhsu[i-1,j]=uc-s.dt*(adv+(s.p[i,j]-s.p[i-1,j])/hx)
    end
    @inbounds for j=2:ny,i=1:nx
        vc=_v(s,i,j)
        vn,vs=_v(s,i,j+1),_v(s,i,j-1)
        ve,vw=_v(s,i+1,j),_v(s,i-1,j)
        ue=(_u(s,i+1,j-1)+_u(s,i+1,j))/2
        uw=(_u(s,i,j-1)+_u(s,i,j))/2
        adv=(_flux(vc,ve,ue,gamma)-_flux(vw,vc,uw,gamma))/hx+
            (_flux(vc,vn,(vc+vn)/2,gamma)-_flux(vs,vc,(vs+vc)/2,gamma))/hy
        s.rhsv[i,j-1]=vc-s.dt*(adv+(s.p[i,j]-s.p[i,j-1])/hy)
    end
    viscosity=(c.lid_velocity==0 ? one(T) : abs(c.lid_velocity))*c.Lx/c.Re
    @views s.rhsu[:,end] .+= 2s.dt*viscosity*c.lid_velocity/hy^2
    # Work arrays hold the previous velocity for the change diagnostic after diffusion.
    old_u,old_v=s.previous_u,s.previous_v
    copyto!(old_u,s.u);copyto!(old_v,s.v)
    solve!(view(s.u,2:nx,:),s.rhsu,s.diffusion_u;reltol=100eps(T),store_history=false)
    solve!(view(s.v,:,2:ny),s.rhsv,s.diffusion_v;reltol=100eps(T),store_history=false)
    _divergence!(s.rhsp,s)
    s.divergence_before=norm(s.rhsp)/sqrt(length(s.rhsp))
    s.rhsp ./= -s.dt
    _recenter!(s.rhsp)
    result=solve!(s.pressure_increment,s.rhsp,s.pressure;reltol=c.pressure_reltol,
                  abstol=100eps(T),maxiter=c.pressure_maxiter,store_history=false)
    result.converged || throw(ErrorException("pressure solver failed to converge: residual $(result.residual_norm)"))
    @inbounds for j=1:ny,i=2:nx
        s.u[i,j]-=s.dt*(s.pressure_increment[i,j]-s.pressure_increment[i-1,j])/hx
    end
    @inbounds for j=2:ny,i=1:nx
        s.v[i,j]-=s.dt*(s.pressure_increment[i,j]-s.pressure_increment[i,j-1])/hy
    end
    s.p .+= s.pressure_increment
    _recenter!(s.p)
    s.divergence_after=divergence_norm(s)
    change=zero(T)
    @inbounds for i in eachindex(s.u)
        change+=(s.u[i]-old_u[i])^2
    end
    @inbounds for i in eachindex(s.v)
        change+=(s.v[i]-old_v[i])^2
    end
    s.velocity_change=sqrt(change/(length(s.u)+length(s.v)))/s.dt
    s.steps+=1
    s.time=s.steps==ceil(Int,c.tf/c.dt) ? c.tf : s.steps*s.dt
    result
end

"""
    CavityResult

Simulation output: final `state`, sampled `times`, RMS divergence `before` and
`after` each projection, and RMS velocity time derivative `velocity_change`.
The `steady` flag reports whether the final change meets `steady_tol`.
"""
struct CavityResult{T,S}
    state::S
    times::Vector{T}
    before::Vector{T}
    after::Vector{T}
    velocity_change::Vector{T}
    steady::Bool
end

"""
    simulate_cavity(config=CavityConfig(); callback=nothing, steady_tol=0)

Run without loading graphics and return [`CavityResult`](@ref). An optional
`callback(state)` runs after each step; it can collect diagnostics or update a plot.
A positive `steady_tol` permits stopping when the RMS velocity time derivative
falls below it (after at least ten steps). Zero runs to the configured final time.
"""
function simulate_cavity(c::CavityConfig{T}=CavityConfig();callback=nothing,steady_tol=0) where T
    isfinite(steady_tol) && steady_tol>=0 || throw(ArgumentError("steady_tol must be finite and nonnegative"))
    s=CavityState(c)
    times,before,after,change=T[],T[],T[],T[]
    for _=1:ceil(Int,c.tf/c.dt)
        step!(s)
        push!(times,s.time);push!(before,s.divergence_before);push!(after,s.divergence_after)
        push!(change,s.velocity_change)
        callback===nothing || callback(s)
        steady_tol>0 && s.steps>=10 && s.velocity_change<=steady_tol && break
    end
    CavityResult(s,times,before,after,change,s.velocity_change<=steady_tol)
end

"""
    centerline_velocities(state)

Return `(y, u, x, v)` for the vertical and horizontal cavity centerlines.
Velocities are interpolated in the transverse direction on staggered faces and
include prescribed boundary values. Coordinates are in physical domain units.
"""
function centerline_velocities(s::CavityState{T}) where T
    c=s.config
    function midline(a,index,axis)
        lo=clamp(floor(Int,index),1,size(a,axis));hi=clamp(ceil(Int,index),1,size(a,axis))
        w=T(index-floor(index))
        axis==1 ? (1-w).*a[lo,:].+w.*a[hi,:] : (1-w).*a[:,lo].+w.*a[:,hi]
    end
    u=midline(s.u,c.nx/2+1,1);v=midline(s.v,c.ny/2+1,2)
    y=vcat(zero(T),collect(((1:c.ny).-T(0.5)).*(c.Ly/c.ny)),c.Ly)
    x=vcat(zero(T),collect(((1:c.nx).-T(0.5)).*(c.Lx/c.nx)),c.Lx)
    (y=y,u=vcat(zero(T),u,c.lid_velocity),x=x,v=vcat(zero(T),v,zero(T)))
end
