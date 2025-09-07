using Test
using LidJul
using LinearAlgebra
using SparseArrays

@testset "LidJul.jl" begin
    @testset "GMG Solver Regression" begin
        n = 16 # Use a small grid for the test
        l2d = LidJul.Laplacian2D(n, n, 1.0, 1.0, dirichlet, dirichlet, dirichlet, dirichlet)
        gmg = LidJul.PoissonGMG(l2d, LidJul.GSSmoother)

        # Method of manufactured solutions
        # u(x,y) = sin(πx)sin(πy)
        # f(x,y) = -Δu = 2π²sin(πx)sin(πy)

        x = range(0, 1, length=n)
        y = range(0, 1, length=n)

        # RHS
        f = [2 * π^2 * sin(π * xi) * sin(π * yi) for xi in x, yi in y]

        # Initial guess (zero)
        u_initial = zeros(n, n)

        # Solve
        LidJul.solve!(u_initial, f, gmg)

        # Analytical solution
        u_analytical = [sin(π * xi) * sin(π * yi) for xi in x, yi in y]

        # Check that the numerical solution is close to the analytical one
        @test u_initial ≈ u_analytical atol=1e-3
    end

    @testset "Laplacian2D" begin
        @testset "allneumann constructor" begin
            l2d_nnnn = LidJul.Laplacian2D(10, 10, 1.0, 1.0, neumann, neumann, neumann, neumann)
            @test l2d_nnnn.allneumann == true

            l2d_dddd = LidJul.Laplacian2D(10, 10, 1.0, 1.0, dirichlet, dirichlet, dirichlet, dirichlet)
            @test l2d_dddd.allneumann == false

            l2d_dndn = LidJul.Laplacian2D(10, 10, 1.0, 1.0, dirichlet, neumann, dirichlet, neumann)
            @test l2d_dndn.allneumann == false
        end

        @testset "sparse_corr" begin
            # This test will check the fix for the sparse_corr function.
            # I will create an all-Neumann case and check that the correction is applied.
            l2d_nnnn = LidJul.Laplacian2D(10, 10, 1.0, 1.0, neumann, neumann, neumann, neumann)
            sp_no_corr = sparse(l2d_nnnn)
            sp_corr = LidJul.sparse_corr(l2d_nnnn)
            @test sp_corr[1,1] ≈ (3/2) * sp_no_corr[1,1]
        end
    end

    @testset "PoissonGMG" begin
        @testset "interpolate_correct" begin
            # This test will check the fix for the interpolation function.
            # I will create a simple coarse grid and check that the interpolation
            # to the fine grid is correct.
            gmg = LidJul.PoissonGMG(LidJul.Laplacian2D(8, 8, 1.0, 1.0, dirichlet, dirichlet, dirichlet, dirichlet), LidJul.GSSmoother)

            # coarse grid is level 2, size 2x2 interior
            # fine grid is level 1, size 4x4 interior
            uc = gmg.Sol[2]
            uf = gmg.Sol[1]

            # Set a value on the coarse grid
            fill!(uc, 0.0)
            uc[3,3] = 1.0 # interior point (2,2) on a 4x4 grid with ghosts

            # Clear the fine grid
            fill!(uf, 0.0)

            # Call the function to test
            LidJul.interpolate_correct(2, gmg)

            # Check the interpolated values on the fine grid
            # The four fine points corresponding to the coarse point uc[3,3] are uf[5,5], uf[4,5], uf[5,4], uf[4,4]
            # with ghost layers, fi=2*3-2=4, fj=2*3-2=4
            @test uf[4,4] ≈ 1.0
            @test uf[5,4] ≈ 0.5
            @test uf[4,5] ≈ 0.5
            @test uf[5,5] ≈ 0.25
        end
    end
end
