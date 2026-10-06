using LidJul, BenchmarkTools, SparseArrays, LinearAlgebra, Pkg, TOML, Dates

const SOLVER_FACTORIES=("Tensor"=>PoissonTTSolver,"SparseLU"=>PoissonSparseLU,
    "AMG"=>PoissonSparseAMG,"ILU_GMRES"=>PoissonSparseCGILU,
    "CG"=>PoissonSparseIterative,"GMG"=>PoissonGMG)
const BOUNDARIES=("DDDD"=>(dirichlet,dirichlet,dirichlet,dirichlet),
    "DNDN"=>(dirichlet,neumann,dirichlet,neumann),
    "NNNN"=>(neumann,neumann,neumann,neumann))

"""Create separate setup/solve trials and accuracy metadata for every case."""
function benchmark_suite(;grids=((16,32),(32,64),(64,64)),types=(Float32,Float64),
                         tolerances=nothing,maxiter=1000)
    suite=BenchmarkGroup();metadata=Dict{String,Any}()
    for T in types,shape in grids,(boundary,bc) in BOUNDARIES
        lap=Laplacian2D(shape...,1,1.5,bc...;T)
        a=sparse(lap)
        exact=T[sin(.1i+.2j) for i=1:shape[1],j=1:shape[2]]
        b=reshape(a*vec(exact),shape)
        rtols=tolerances===nothing ? (T===Float32 ? (1e-2,1e-3) : (1e-6,1e-8)) : tolerances
        for (name,factory) in SOLVER_FACTORIES,reltol in rtols
            label="$T/$(shape[1])x$(shape[2])/$boundary/$name/$reltol"
            solver=factory(lap);x=zeros(T,shape)
            result=solve!(x,b,solver;reltol,maxiter,store_history=false)
            result.converged || error("accuracy check failed for $label: $(result.residual_norm/norm(b))")
            group=suite[label]=BenchmarkGroup()
            group["setup"]=@benchmarkable $factory($lap) evals=1
            # Every solve sample starts from zero; repeated evaluations would time a solved RHS.
            group["solve"]=@benchmarkable solve!($x,$b,$solver;reltol=$reltol,maxiter=$maxiter,store_history=false) setup=(fill!($x,zero($T))) evals=1
            metadata[label]=Dict("nx"=>shape[1],"ny"=>shape[2],"scalar_type"=>string(T),
                "boundary"=>boundary,"solver"=>name,"reltol"=>reltol,"abstol"=>0.0,"converged"=>result.converged,"maxiter"=>maxiter,
                "iterations"=>result.iterations,"relative_residual"=>Float64(result.residual_norm/norm(b)))
        end
    end
    suite,metadata
end

"""Benchmark and write complete setup, solve, allocation and environment metadata."""
function benchmark_report(;output=joinpath(@__DIR__,"results.toml"),seconds=.1,samples=20,kwargs...)
    suite,metadata=benchmark_suite(;kwargs...)
    results=run(suite;seconds,samples,verbose=false)
    cases=Dict{String,Any}[]
    for label in sort(collect(keys(metadata)))
        row=copy(metadata[label])
        for phase in ("setup","solve")
            estimate=minimum(results[label][phase])
            row[phase*"_nanoseconds"]=estimate.time
            row[phase*"_bytes"]=estimate.memory
            row[phase*"_allocations"]=estimate.allocs
            row[phase*"_samples"]=length(results[label][phase])
        end
        push!(cases,row)
    end
    versions=Dict(info.name=>string(info.version) for info in values(Pkg.dependencies()) if info.version!==nothing)
    loaded_versions=Dict(id.name=>string(Base.pkgversion(mod)) for (id,mod) in Base.loaded_modules
                         if Base.pkgversion(mod)!==nothing)
    report=Dict("julia"=>string(VERSION),"generated_utc"=>string(now(UTC)),
                "cpu"=>Sys.CPU_NAME,"architecture"=>string(Sys.ARCH),"kernel"=>string(Sys.KERNEL),
                "julia_threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),
                "blas"=>sprint(show,BLAS.get_config()),"timing_statistic"=>"minimum warmed trial; evals=1",
                "versions"=>versions,"loaded_versions"=>loaded_versions,"cases"=>cases)
    open(io->TOML.print(io,report),output,"w")
    report
end
