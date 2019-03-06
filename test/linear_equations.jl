using Test
import LinearAlgebra: Diagonal, LowerTriangular, UpperTriangular
using .LinearEquations

# A 为系数矩阵, b 为常数项, x 为迭代初值, 第二个字母指代类型.
A = Ar = Rational{Int}[
    2   1   0;
    0   2   1;
    1   0   2;
]
Ai = Ar .|> Rational
Af = Ar .|> Float64
Ac = Ar .|> ComplexF64
b = br = Rational{Int}[2, 1, 0]
bi = br .|> Int
bf = br .|> Float64
bc = br .|> ComplexF64
x = xr = Rational{Int}[0, 1, 2]
xi = xr .|> Int
xf = xr .|> Float64
xc = xr .|> ComplexF64

@testset "Iterative Method" begin
    # D, L, U 分别为系数矩阵的对角, 下三角与上三角部分, 且 L, U 对角线全为 0.
    D = Diagonal(A)
    L = LowerTriangular(A) - D
    U = UpperTriangular(A) - D

    @testset "Jacobi Method" begin
        B = - D^(-1) * (L + U)
        f = D^(-1) * b
        itr = gen_itr(A, b, Jacobi)
        @test (x |> itr) == B * x + f
        @test (x |> itr |> itr) == B * (B * x + f) + f
        @test (x |> itr |> itr |> itr) == B * (B * (B * x + f) + f) + f
        @test solve(Af, bf, Jacobi) ≈ A \ b  atol = 1e-10
        @test_nowarn for A in [Ar, Ai, Af, Ac], b in [br, bi, bf, bc], x in [xr, xi, xf, xc]
            x |> gen_itr(A, b, Jacobi)
        end
    end

    @testset "GaussSeidel Method" begin
        B = - (D + L)^(-1) * U
        f = (D + L)^(-1) * b
        itr = gen_itr(A, b, GaussSeidel)
        @test (x |> itr) == B * x + f
        @test (x |> itr |> itr) == B * (B * x + f) + f
        @test (x |> itr |> itr |> itr) == B * (B * (B * x + f) + f) + f
        @test solve(Af, bf, GaussSeidel) ≈ A \ b  atol = 1e-10
        @test_nowarn for A in [Ar, Ai, Af, Ac], b in [br, bi, bf, bc], x in [xr, xi, xf, xc]
            x |> gen_itr(A, b, GaussSeidel)
        end
    end

    @testset "SOR Method" begin
        ω = 9 // 10
        B = (D + ω * L)^(-1) * ((1 - ω) * D - ω * U)
        f = ω * (D + ω * L)^(-1) * b
        itr = gen_itr(A, b, SOR, ω)
        @test (x |> itr) == B * x + f
        @test (x |> itr |> itr) == B * (B * x + f) + f
        @test (x |> itr |> itr |> itr) == B * (B * (B * x + f) + f) + f
        @test solve(Af, bf, SOR, x, ω) ≈ A \ b  atol = 1e-10
        @test_nowarn for A in [Ar, Ai, Af, Ac], b in [br, bi, bf, bc], x in [xr, xi, xf, xc]
            x |> gen_itr(A, b, SOR, ω)
        end
    end
end
