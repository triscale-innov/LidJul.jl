@testset "Solver contract, scalar types and rectangular grids" begin
    D,N=dirichlet,neumann
    factories=(PoissonTTSolver,PoissonSparseLU,PoissonSparseAMG,PoissonSparseCGILU,PoissonSparseIterative,PoissonGMG)
    for T in (Float32,Float64), shape in ((8,16),(16,8)),
        bc in ((D,D,D,D),(D,N,D,N),(N,D,N,D),(N,N,N,N))
        lap=Laplacian2D(shape...,1,1.5,bc...;T)
        a=sparse(lap)
        exact=T[sin(0.3i+0.7j) for i=1:shape[1],j=1:shape[2]]
        lap.allneumann && (exact .-= sum(exact)/length(exact))
        rhs=reshape(a*vec(exact),shape)
        tolerance=T===Float32 ? T(2e-4) : T(1e-8)
        for factory in factories
            s=factory(lap)
            @test all(t->t!==Any,fieldtypes(typeof(s)))
            x=zeros(T,shape)
            result=solve!(x,rhs,s;reltol=tolerance,maxiter=1000)
            @test result isa SolveResult{T}
            @test result.converged
            @test result.residual_norm≈norm(a*vec(x)-vec(rhs)) rtol=1e-2 atol=20eps(T)*norm(rhs)
            @test result.residual_norm<=tolerance*norm(rhs)
            @test norm(x-exact)/norm(exact)<(T===Float32 ? 0.015 : 1e-5)
            @test !isempty(result.history) && result.history[end]==result.residual_norm
            lap.allneumann && @test abs(sum(x)/length(x))<20eps(T)
            # A warm start must obey a tolerance relative to the RHS, not its initial residual.
            x .= exact .+ T(0.01)
            result=solve!(x,rhs,s;reltol=tolerance,store_history=false)
            @test result.converged && isempty(result.history)
            @test_throws DimensionMismatch solve!(zeros(T,2,2),rhs,s)
            @test_throws ArgumentError solve!(x,rhs,s;reltol=-1)
            @test_throws ArgumentError solve!(x,rhs,s;maxiter=-1)
            @test_throws ArgumentError solve!(x,x,s)
            @test_throws ArgumentError solve!(zeros(T===Float32 ? Float64 : Float32,shape),rhs,s)
            fill!(x,0)
            zero_iterations=solve!(x,rhs,s;maxiter=0,reltol=tolerance)
            @test !zero_iterations.converged && zero_iterations.iterations==0
            if lap.allneumann
                @test_throws ArgumentError solve!(x,ones(T,shape),s)
                @test_throws ArgumentError solve!(x,fill(T(1e-20),shape),s)
            end
        end
    end
    for T in (Float32,Float64),bc in Iterators.product((D,N),(D,N),(D,N),(D,N))
        lap=Laplacian2D(4,8,1,2,bc...;T)
        exact=T[cos(0.2i+0.5j) for i=1:4,j=1:8]
        lap.allneumann && (exact .-= sum(exact)/length(exact))
        b=reshape(sparse(lap)*vec(exact),4,8)
        for s in (PoissonTTSolver(lap),PoissonGMG(lap))
            x=zeros(T,4,8)
            @test solve!(x,b,s;reltol=T===Float32 ? 2e-4 : 1e-8).converged
        end
    end
    @test maxlevels(16)==4
    @test maxlevels(8,32)==3
    @test_throws ArgumentError maxlevels(6)
    @test_throws ArgumentError maxlevels(3)
    lap=Laplacian2D(8,8,1,1,D,D,D,D)
    @test PoissonGMG_new(lap) isa PoissonGMG
    @test solve!(zeros(8,8),ones(8,8),PoissonGMG(lap;nlevels=1)).converged
    @test_throws ArgumentError PoissonGMG(lap;nlevels=0)
    @test_throws ArgumentError PoissonGMG(lap;nprae=0,npost=0)
    @test_throws ArgumentError solve!(zeros(8,8),ones(8,8),PoissonSparseIterative(lap);method="unknown")
    for bc in ((D,D,D,D),(N,N,N,N)), method in ("jacobi","gauss_seidel","sor","ssor")
        lap=Laplacian2D(4,4,1,1,bc...)
        exact=[sin(0.3i+0.2j) for i=1:4,j=1:4]
        b=reshape(sparse(lap)*vec(exact),4,4)
        x=zeros(4,4)
        @test solve!(x,b,PoissonSparseIterative(lap);method,reltol=1e-7,maxiter=3000).converged
    end
    # Helmholtz solve with different boundary discretizations and spacings.
    nx,ny=5,7
    ax=0.1LidJul.K1(nx,0.2,2);ay=0.2LidJul.K1(ny,0.3,3)
    s=TensorialOperator(ax,ay;shift=1)
    a=kron(sparse(I,ny,ny),sparse(ax))+kron(sparse(ay),sparse(I,nx,nx))+I
    exact=[sin(0.2i+0.8j) for i=1:nx,j=1:ny]
    b=reshape(a*vec(exact),nx,ny);x=zeros(nx,ny)
    @test solve!(x,b,s;reltol=1e-10).converged
    @test x≈exact rtol=1e-10
end

@testset "Second-order Poisson accuracy" begin
    errors=Float64[]
    for n in (12,24,48)
        lap=Laplacian2D(n,2n,1,2,dirichlet,dirichlet,dirichlet,dirichlet)
        exact=[sinpi((i-0.5)/n)*sinpi((j-0.5)/(2n)) for i=1:n,j=1:2n]
        rhs=(π^2+(π/2)^2).*exact;x=zeros(n,2n)
        solve!(x,rhs,PoissonTTSolver(lap))
        push!(errors,norm(x-exact)/norm(exact))
    end
    @test 3.8<errors[1]/errors[2]<4.2
    @test 3.8<errors[2]/errors[3]<4.2
end

@testset "Bounded cached solve allocations and independent workspaces" begin
    function allocated_solve(n,factory)
        lap=Laplacian2D(n,2n,1,2,dirichlet,dirichlet,dirichlet,dirichlet)
        s=factory(lap);b=ones(n,2n);x=zeros(n,2n)
        solve!(x,b,s;reltol=1e-7,store_history=false)
        fill!(x,0)
        @allocated solve!(x,b,s;reltol=1e-7,store_history=false)
    end
    for f in (PoissonTTSolver,PoissonSparseLU,PoissonGMG)
        small,large=allocated_solve(8,f),allocated_solve(32,f)
        # Workspace reuse must not allocate a fresh full grid per solve.
        @test large<small+8192
    end
    lap=Laplacian2D(8,16,1,2,dirichlet,dirichlet,dirichlet,dirichlet)
    tasks=[Threads.@spawn begin
        x=zeros(8,16);s=PoissonTTSolver(lap)
        r=solve!(x,fill(Float64(k),8,16),s;reltol=1e-10)
        (x,r)
    end for k=1:2]
    outcomes=fetch.(tasks)
    @test all(o->o[2].converged,outcomes)
    @test outcomes[2][1]≈2outcomes[1][1]
end
