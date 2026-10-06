@testset "Cavity projection, boundaries and parameters" begin
    for T in (Float32,Float64),solver in (:tensor,:gmg,:lu,:amg)
        c=CavityConfig(;nx=8,ny=16,Lx=1,Ly=1.5,tf=0.1,dt=0.01,T,pressure_solver=solver)
        r=simulate_cavity(c)
        @test r.state.time==c.tf
        @test r.state.steps==10
        @test size(r.state.u)==(9,16)
        @test size(r.state.v)==(8,17)
        @test maximum(r.after)<(T===Float32 ? 2e-5 : 2e-8)
        @test all(r.after .< r.before)
        @test wall_error(r.state)<10eps(T)
        @test abs(sum(r.state.p)/length(r.state.p))<10eps(T)
        @test all(isfinite,r.state.u) && all(isfinite,r.state.v)
        @test_throws ArgumentError step!(r.state)
        lines=centerline_velocities(r.state)
        @test lines.u[[1,end]]==[0,1]
        @test lines.v[[1,end]]==[0,0]
    end
    c=CavityConfig(;nx=7,ny=9,dt=.03,tf=.1,lid_velocity=0)
    called=Ref(0)
    r=simulate_cavity(c;callback=s->(called[]+=1))
    @test called[]==4 && r.state.time==c.tf
    @test iszero(norm(r.state.u))+iszero(norm(r.state.v))==2
    @test divergence_norm(r.state)==0 && wall_error(r.state)==0
    @test_throws ArgumentError CavityConfig(;Re=0)
    @test_throws ArgumentError CavityConfig(;dt=-1)
    @test_throws ArgumentError CavityConfig(;donor_cell=2)
    @test_throws ArgumentError CavityConfig(;nx=6,pressure_solver=:gmg)
    @test_throws ArgumentError CavityConfig(;pressure_solver=:unknown)
    @test_throws ArgumentError simulate_cavity(CavityConfig(;);steady_tol=-1)
    @test_throws ArgumentError step!(CavityState(CavityConfig(;nx=8,dt=1,tf=1)))
end

@testset "Cavity timestep refinement" begin
    states=[simulate_cavity(CavityConfig(;nx=12,ny=16,tf=.4,dt=dt)).state for dt in (.02,.01,.005)]
    errors=[norm(states[k].u-states[3].u)+norm(states[k].v-states[3].v) for k=1:2]
    @test errors[2]<errors[1]/2
    @test all(s->divergence_norm(s)<1e-10,states)
end

@testset "Incremental projection preserves the steady velocity under timestep refinement" begin
    results=[simulate_cavity(CavityConfig(;nx=8,tf=30,dt,pressure_reltol=1e-11);steady_tol=1e-7)
             for dt in (.02,.01)]
    @test all(r->r.steady,results)
    @test norm(results[1].state.u-results[2].state.u)<1e-6
    @test norm(results[1].state.v-results[2].state.v)<1e-6
end
