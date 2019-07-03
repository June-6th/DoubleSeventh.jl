module LinearEquations

using LinearAlgebra: diag, norm, diagind

export
    solve,
    # 迭代法类型及其实例
    IterativeMethod,
    JacobiMethod, Jacobi,
    GaussSeidelMethod, GaussSeidel,
    SORMethod, SOR,
    # 迭代法相关函数
    multiple_itr, itr_func

"""
    IterativeMethod

所有迭代法的抽象类型.
"""
abstract type IterativeMethod end

struct JacobiMethod <: IterativeMethod end
struct GaussSeidelMethod <: IterativeMethod end
struct SORMethod <: IterativeMethod end

@doc raw"""
    Jacobi

给定 ``n`` 阶线性方程组 ``Ax = b``, 设 ``A = (a_{ij}), b = (b_{i})``,
其中 ``1 \leqslant i, j \leqslant n`` 且 ``A`` 的对角元都不为 ``0``.
``D, L, U`` 分别为 ``A`` 的主对角, 上三角和下三角部分, 即:

```math
D = \begin{bmatrix}
    a_{11} & 0      & \cdots & 0 \\
    0      & a_{22} & \cdots & 0 \\
    \vdots & \vdots & \ddots & \vdots \\
    0      & 0      & \cdots & a_{nn}
\end{bmatrix},
L = \begin{bmatrix}
    0      & a_{12} & \cdots & a_{1n} \\
    0      & 0      & \cdots & a_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    0      & 0      & \cdots & 0
\end{bmatrix},
U = \begin{bmatrix}
    0      & 0      & \cdots & 0 \\
    a_{21} & 0      & \cdots & 0 \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{n1} & a_{n2} & \cdots & 0
\end{bmatrix},
```

``x^{(k)}, x^{(k + 1)}`` 分别为第 ``k`` 次及第 ``k + 1`` 次迭代后得到的值,
则 Jacobi 迭代法的迭代公式为:

```math
\begin{cases}
    x_1^{(k + 1)} & = (- a_{12} x_{2}^{(k)} - a_{13} x_{3}^{(k)} \cdots
                      - a_{1n} x_{n}^{(k)}) / a_{11} + b_{1} / a_{11} \\
    x_2^{(k + 1)} & = (- a_{21} x_{1}^{(k)} - a_{23} x_{3}^{(k)} \cdots
                      - a_{2n} x_{n}^{(k)}) / a_{22} + b_{2} / a_{22} \\
                  & \vdots \\
    x_n^{(k + 1)} & = (- a_{n1} x_{1}^{(k)} - a_{n2} x_{2}^{(k)} \cdots
                      - a_{nn - 1} x^{(k)}) / a_{nn} + b_{n} / a_{nn} \\
\end{cases}
```

其矩阵形式为 ``x^{(k + 1)} = - D^{-1} (L + U) x^{(k)} + D^{-1} b``,
故 Jacobi 迭代法的迭代矩阵为 ``B_{J} = - D^{-1} (L + U)``, 常数项为 ``f = D^{-1} b``,
"""
const Jacobi = JacobiMethod()
@doc raw"""
    GaussSeidel

给定 ``n`` 阶线性方程组 ``Ax = b``, 设 ``A = (a_{ij}), b = (b_{i})``,
其中 ``1 \leqslant i, j \leqslant n`` 且 ``A`` 的对角元都不为 ``0``.
``D, L, U`` 分别为 ``A`` 的主对角, 上三角和下三角部分, 即:

```math
D = \begin{bmatrix}
    a_{11} & 0      & \cdots & 0 \\
    0      & a_{22} & \cdots & 0 \\
    \vdots & \vdots & \ddots & \vdots \\
    0      & 0      & \cdots & a_{nn}
\end{bmatrix},
L = \begin{bmatrix}
    0      & a_{12} & \cdots & a_{1n} \\
    0      & 0      & \cdots & a_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    0      & 0      & \cdots & 0
\end{bmatrix},
U = \begin{bmatrix}
    0      & 0      & \cdots & 0 \\
    a_{21} & 0      & \cdots & 0 \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{n1} & a_{n2} & \cdots & 0
\end{bmatrix},
```

``x^{(k)}, x^{(k + 1)}`` 分别为第 ``k`` 次及第 ``k + 1`` 次迭代后得到的值,
则 GaussSeidel 迭代法的迭代公式为:

```math
\begin{cases}
    x_1^{(k + 1)} & = (- a_{12} x_{2}^{(k)} - a_{13} x_{3}^{(k)} \cdots
                      - a_{1n} x_{n}^{(k)}) / a_{11} + b_{1} / a_{11} \\
    x_2^{(k + 1)} & = (- a_{21} x_{1}^{(k + 1)} - a_{23} x_{3}^{(k)} \cdots
                      - a_{2n} x_{n}^{(k)}) / a_{22} + b_{2} / a_{22} \\
                  & \vdots \\
    x_n^{(k + 1)} & = (- a_{n1} x_{1}^{(k + 1)} - a_{n2} x_{2}^{(k + 1)} \cdots
                      - a_{nn - 1} x^{(k + 1)}) / a_{nn} + b_{n} / a_{nn} \\
\end{cases}
```

将 ``x^{(k + 1)}`` 的分量移至左边并同乘以 ``D`` 后,
可得其矩阵形式为 ``(D + L) x^{(k + 1)} = U x^{(k)} + b``,
于是 ``x^{(k + 1)} = - (D + L)^{-1} U x^{(k)} + (D + L)^{-1} b``,
故 GaussSeidel 迭代法的迭代矩阵为 ``B_{GS} = - (D + L)^{-1} U``,
常数项为 ``f = (D + L)^{-1} b``.
"""
const GaussSeidel = GaussSeidelMethod()
@doc raw"""
    SOR

给定 ``n`` 阶线性方程组 ``Ax = b``, 设 ``A = (a_{ij}), b = (b_{i})``,
其中 ``1 \leqslant i, j \leqslant n`` 且 ``A`` 的对角元都不为 ``0``.
``D, L, U`` 分别为 ``A`` 的主对角, 上三角和下三角部分, 即:

```math
D = \begin{bmatrix}
    a_{11} & 0      & \cdots & 0 \\
    0      & a_{22} & \cdots & 0 \\
    \vdots & \vdots & \ddots & \vdots \\
    0      & 0      & \cdots & a_{nn}
\end{bmatrix},
L = \begin{bmatrix}
    0      & a_{12} & \cdots & a_{1n} \\
    0      & 0      & \cdots & a_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    0      & 0      & \cdots & 0
\end{bmatrix},
U = \begin{bmatrix}
    0      & 0      & \cdots & 0 \\
    a_{21} & 0      & \cdots & 0 \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{n1} & a_{n2} & \cdots & 0
\end{bmatrix},
```

``x^{(k)}, x^{(k + 1)}`` 分别为第 ``k`` 次及第 ``k + 1`` 次迭代后得到的值,
``\tilde{x}^{(k + 1)}`` 是 [`GaussSeidel`](@ref) 法定义的中间变量,
``\omega`` 为松弛因子, 则 SOR 迭代法的迭代公式为:

```math
\begin{cases}
    \tilde{x}_1^{(k + 1)} & = (- a_{12} x_{2}^{(k)} - a_{13} x_{3}^{(k)} \cdots
                              - a_{1n} x_{n}^{(k)}) / a_{11} + b_{1} / a_{11} \\
    \tilde{x}_2^{(k + 1)} & = (- a_{21} x_{1}^{(k + 1)} - a_{23} x_{3}^{(k)} \cdots
                              - a_{2n} x_{n}^{(k)}) / a_{22} + b_{2} / a_{22} \\
                          & \vdots \\
    \tilde{x}_n^{(k + 1)} & = (- a_{n1} x_{1}^{(k + 1)} - a_{n2} x_{2}^{(k + 1)} \cdots
                              - a_{nn - 1} x^{(k + 1)}) / a_{nn} + b_{n} / a_{nn} \\
    x^{(k + 1)}           & = (1 - \omega) x^{(k)} + \omega \tilde{x}^{(k + 1)}
\end{cases}
```

其矩阵形式为:

```math
\begin{cases}
    \tilde{x}^{(k + 1)} & = - D^{-1} L x^{(k + 1)} - D^{-1} U x^{(k)} + D^{-1} b \\
    x^{(k + 1)}         & = (1 - \omega) x^{(k)} + \omega \tilde{x}^{(k + 1)}
\end{cases}
```

代入并化简得:

```math
x^{(k + 1)} = (D + \omega L)^{-1} ((1 - \omega) D - \omega U) x^{(k)}
              + \omega (D + \omega L)^{-1} b
```

故 SOR 迭代法的迭代矩阵为 ``B_{SOR} = (D + \omega L)^{-1} ((1 - \omega) D - \omega U)``,
常数项为 ``f = \omega (D + \omega L)^{-1} b``.
"""
const SOR = SORMethod()

"""
    itr_func(A, b, alg, args...; kwargs...)

生成线性方程组 `Ax=b` 在迭代算法 `alg` 下的迭代函数.

# Implementation
设所生成的迭代函数为 `itr`, 迭代初值为 `x`, 则 `itr(x)` 返回 `x` 进行一次迭代后生成的值,
`args` 和 `kwargs` 为迭代算法 `alg` 的参数, 如 [`SOR`](@ref) 迭代法中的松弛因子 `ω`.
"""
function itr_func end

function itr_func(A::AbstractMatrix, b::AbstractVector, alg::JacobiMethod)::Function
    (axes(A, 1) == axes(A, 2) && axes(A, 2) == axes(b, 1)) || throw(DimensionMismatch("维度不匹配"))
    d = one(promote_type(eltype(A), eltype(b))) ./ diag(A)
    B = @. - A * d; B[diagind(B)] .= zero(eltype(B))
    f = d; @. f = b * d
    return function itr(x::AbstractVector)::AbstractVector
        axes(x) == axes(f) || throw(DimensionMismatch("维度不匹配"))
        y = similar(x, promote_type(eltype(x), eltype(f))); @. y = f
        for j in axes(A, 2)
            for i in axes(A, 1)
                @inbounds y[i] += B[i, j] * x[j]
            end
        end
        return y
    end
end

function itr_func(A::AbstractMatrix, b::AbstractVector, alg::GaussSeidelMethod)::Function
    (axes(A, 1) == axes(A, 2) && axes(A, 2) == axes(b, 1)) || throw(DimensionMismatch("维度不匹配"))
    d = one(promote_type(eltype(A), eltype(b))) ./ diag(A)
    B = @. - A * d
    f = d; @. f = b * d
    return function itr(x::AbstractVector)::AbstractVector
        axes(x) == axes(f) || throw(DimensionMismatch("维度不匹配"))
        y = similar(x, promote_type(eltype(x), eltype(f))); @. y = f
        for j in (firstindex(B, 2) + 1):lastindex(B, 2)
            for i in firstindex(B, 1):(j - 1)
                @inbounds y[i] += B[i, j] * x[j]
            end
        end
        for j in axes(B, 2)
            for i in (j + 1):lastindex(B, 2)
                @inbounds y[i] += B[i, j] * y[j]
            end
        end
        return y
    end
end

"""
    itr_func(A, b, alg::SORMethod, ω)

`ω` 为 [`SOR`](@ref) 法中的松弛因子.
"""
function itr_func(A::AbstractMatrix, b::AbstractVector, alg::SORMethod, ω::Real)::Function
    (axes(A, 1) == axes(A, 2) && axes(A, 2) == axes(b, 1)) || throw(DimensionMismatch("维度不匹配"))
    d = one(promote_type(eltype(A), eltype(b))) ./ diag(A)
    B = @. - ω * A * d
    f = d; @. f = b * d
    return function itr(x::AbstractVector)::AbstractVector
        axes(x) == axes(f) || throw(DimensionMismatch("维度不匹配"))
        y = @. (1 - ω) * x + ω * f
        for j in (firstindex(B, 2) + 1):lastindex(B, 2)
            for i in firstindex(B, 1):(j - 1)
                @inbounds y[i] += B[i, j] * x[j]
            end
        end
        for j in axes(B, 2)
            for i in (j + 1):lastindex(B, 2)
                @inbounds y[i] += B[i, j] * y[j]
            end
        end
        return y
    end
end

"""
    multiple_itr(itr, x₀, args...; n=1000, ϵ=0, kwargs...)

使用迭代函数 `itr` 多次迭代 `x₀`, 返回 `(y, itr_times, norm(x - y))`,
其中 `y` 为迭代值, `itr_times` 为实际迭代次数, `x` 为最后一次迭代的迭代初值,
`norm(x - y)` 为最后一次迭代中迭代初值减迭代值的 2 阶范数.

# Arguments
- `x₀::AbstractVector`: 迭代初值;
- `n::Integer=1000`: 最大迭代次数;
- `ϵ::Real=0`: 在迭代值减迭代初值的 2 阶范数小于 `ϵ` 时中止迭代, 在 `ϵ=0` 时不会中止迭代.
"""
function multiple_itr(itr::Function, x₀; n::Integer=1000, ϵ::Real=0)
    n <= 0 && throw(n, "迭代次数需大于 0")
    x, y = x₀, itr(x₀)
    itr_times = 1
    while (itr_times < n) && (norm(x - y) > ϵ)
        x, y = y, itr(y)
        itr_times += 1
    end
    return y, itr_times, norm(x - y)
end

"""
    solve(A, b, alg::IterativeMethod, x₀=zero(b), args...; n=1000, ϵ=1e-10, kwargs...)

使用迭代算法 `alg` 计算线性方程组 `Ax=b` 的近似解.

# Arguments
- `x₀::AbstractVector=zero(b)`: 迭代初值;
- `n::Integer=1000`: 最大迭代次数;
- `ϵ::Real=1e-10`: 近似控制求解精度, 在迭代中,
  若某次迭代的迭代值减迭代初值的 2 阶范数小于 `ϵ`, 则停止迭代;
  需注意的是, 此时精确解减近似解的 2 阶范数不一定小于 `ϵ`;
- `args`, `kwargs`: 与迭代初值 `x₀` 一同作为迭代算法 `alg` 的参数.
"""
function solve(A::AbstractMatrix, b::AbstractVector, alg::IterativeMethod,
               x₀::AbstractVector=zero(b), args...;
               n::Integer=1000, ϵ::Real=1e-10, kwargs...)
    n <= 0 && throw(n, "迭代次数需大于 0")
    itr = itr_func(A, b, alg, args...; kwargs...)
    x, y = x₀, itr(x₀)
    ϵ₁ = ϵ₂ = norm(x - y)
    itr_times = 1
    while (itr_times < n) && (ϵ₂ > ϵ)
        x, y = y, itr(y)
        ϵ₂ = norm(x - y)
        ϵ₁ < ϵ₂ && throw("迭代法不收敛!") || (ϵ₁ = ϵ₂)
        itr_times += 1
    end
    return y, itr_times, norm(x - y)
end

end # module
