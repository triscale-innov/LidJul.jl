@testset "Laplacian operators" begin
    D, N = dirichlet, neumann
    one = Laplacian1D(5, 2.0, D, N)
    @test size(one) == (5, 5)
    @test size(one, 3) == 1
    @test one.bcl_ == D && one.bcr_ == N
    @test Matrix(one) == Matrix(SymTridiagonal(one))
    @test_throws BoundsError one[0, 1]
    @test_throws BoundsError one[1, 6]
    @test_throws ArgumentError Laplacian1D(1, 1.0, D, D)
    @test_throws ArgumentError Laplacian1D(4, 0.0, D, D)

    # Independent tensor assembly checks rectangular grids and all 16 BCs.
    for nx in (3, 4), ny in (2, 5), bc in Iterators.product((D,N),(D,N),(D,N),(D,N))
        lap = Laplacian2D(nx, ny, 2.0, 3.0, bc...)
        ax, ay = SymTridiagonal(lap.lpx), SymTridiagonal(lap.lpy)
        reference = kron(Matrix{Float64}(I,ny,ny), Matrix(ax)) +
                    kron(Matrix(ay), Matrix{Float64}(I,nx,nx))
        actual = sparse(lap)
        @test Matrix(actual) ≈ reference
        @test issymmetric(actual)
        @test lap.allneumann == all(==(N), bc)
        if lap.allneumann
            @test norm(actual * ones(nx*ny)) < 1e-12
            @test isposdef(Symmetric(Matrix(sparse_corr(lap))))
        else
            @test sparse_corr(lap) == actual
            @test isposdef(Symmetric(Matrix(actual)))
        end
    end
end
