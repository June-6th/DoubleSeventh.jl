module IterativeMethods

using LinearAlgebra: diag, norm, diagind

export solve, multiple_itr, gen_itr, IterativeMethod, Jacobi, JacobiMethod, GaussSeidel, GaussSeidelMethod, SOR, SORMethod

"所有迭代法的抽象类型。"
abstract type IterativeMethod end

"""
    gen_itr(A, b, alg)

生成线性方程组 `Ax=b` 在迭代算法 `alg` 下的迭代函数 `itr`。设迭代初值为 `x₀`，则 `itr(x₀, args...; kwargs...)` 返回 `x₀` 进行一次迭代后生成的值，其中的 `args` 和 `kwargs` 为迭代算法 `alg` 中的参数，如 SOR 迭代法中的松弛因子 `ω`。
"""
function gen_itr end

"""
    multiple_itr(itr, x₀, args...;[n, ϵ, ]kwargs...)

使用迭代函数 `itr` 多次迭代 `x₀`，返回 `(y, itr_times, norm(x - y))`，其中 `y` 为迭代值，`itr_times` 为实际迭代次数，`x` 为最后一次迭代的迭代初值，`norm(x - y)` 为最后一次迭代中迭代初值减迭代值的 2 阶范数。

# Arguments
- `x₀::AbstractVector`：迭代初值；
- `n::Integer=1000`：最大迭代次数；
- `ϵ::Union{Real, Nothing}=nothing`：若为 `nothing` 则不中止迭代，否则在迭代值减迭代初值的 2 阶范数小于 `ϵ` 时中止迭代。
- `args`, `kwargs`：与迭代初值 `x₀` 一同作为迭代函数 `itr` 的参数，迭代中 `itr` 调用格式为 `itr(x₀, args...; kwargs...)`
"""
function multiple_itr(itr::Function, x₀, args...; n::Integer=1000, ϵ::Union{Real, Nothing}=nothing, kwargs...)
    n <= 0 && throw(n, "迭代次数需大于 0！")
    x, y = x₀, itr(x₀, args...; kwargs...)
    itr_times = 1
    while itr_times < n && (isnothing(ϵ) || norm(x - y) > ϵ)
        x, y = y, itr(y, args...; kwargs...)
        itr_times += 1
    end
    y, itr_times, norm(x - y)
end

"""
    solve(A, b, alg[, x₀], args...;[n, ϵ, ]kwargs...)

计算线性方程组 `Ax=b` 的近似解，所用迭代算法为 `alg`。

# Arguments
- `x₀::AbstractVector=zero(b)`：迭代初值；
- `n::Integer=1000`：最大迭代次数；
- `ϵ::Real=1e-10`：近似控制求解精度，在迭代中，若某次迭代的迭代值减迭代初值的 2 阶范数小于 `ϵ` 时停止迭代；需注意的是，此时精确解减近似解的 2 阶范数不一定小于 `ϵ`。
- `args`, `kwargs`：与迭代初值 `x₀` 一同作为迭代函数 `itr` 的参数，其中 `itr = gen_itr(A, b, alg)`，迭代中 `itr` 调用格式为 `itr(x₀, args...; kwargs...)`
"""
function solve(A::AbstractMatrix, b::AbstractVector, alg::IterativeMethod, x₀::AbstractVector=zero(b), args...;n::Integer=1000, ϵ::Real=1e-10, kwargs...)
    n <= 0 && throw(n, "迭代次数需大于 0！")
    itr = gen_itr(A, b, alg)
    multiple_itr(itr, x₀, args...;n=n, ϵ=ϵ, kwargs...)[1]
end

struct JacobiMethod <: IterativeMethod end
const Jacobi = JacobiMethod()

function gen_itr(A::AbstractMatrix, b::AbstractVector, alg::JacobiMethod)::Function
    (axes(A, 1) == axes(A, 2) && axes(A, 2) == axes(b, 1)) || throw(DimensionMismatch("维度不匹配"))
    d = one(promote_type(eltype(A), eltype(b))) ./ diag(A)
    B = @. - A * d; B[diagind(B)] .= zero(eltype(B))
    f = d; @. f = b * d
    function itr(x₀::AbstractVector)::AbstractVector
        y = copy(f)
        @inbounds for j in axes(A, 2)
            for i in axes(A, 1)
                y[i] += B[i, j] * x[j]
            end
        end
        y
    end
end

struct GaussSeidelMethod <: IterativeMethod end
const GaussSeidel = GaussSeidelMethod()

function gen_itr(A::AbstractMatrix, b::AbstractVector, alg::GaussSeidelMethod)::Function
    (axes(A, 1) == axes(A, 2) && axes(A, 2) == axes(b, 1)) || throw(DimensionMismatch("维度不匹配"))
    d = one(promote_type(eltype(A), eltype(b))) ./ diag(A)
    B = @. - A * d
    f = d; @. f = b * d
    function itr(x₀::AbstractVector)::AbstractVector
        y = copy(f)
        @inbounds for j in (firstindex(B, 2) + 1):lastindex(B, 2)
            for i in firstindex(B, 1):(j - 1)
                y[i] += B[i, j] * x[j]
            end
        end
        @inbounds for j in axes(B, 2)
            for i in (j + 1):lastindex(B, 2)
                y[i] += B[i, j] * y[j]
            end
        end
        y
    end
end

struct SORMethod <: IterativeMethod end
const SOR = SORMethod()

"""
    gen_itr(A, b, alg::SORMethod)

SOR 法生成的迭代函数 `itr` 调用形式为 `itr(x₀, ω)`，其中 `ω` 为松弛因子。
"""
function gen_itr(A::AbstractMatrix, b::AbstractVector, alg::SORMethod)::Function
    (axes(A, 1) == axes(A, 2) && axes(A, 2) == axes(b, 1)) || throw(DimensionMismatch("维度不匹配"))
    d = one(promote_type(eltype(A), eltype(b))) ./ diag(A)
    B = @. - A * d
    f = d; @. f = b * d
    function itr(x₀::AbstractVector, ω::Real)::AbstractVector
        y = copy(f)
        @inbounds for j in (firstindex(B, 2) + 1):lastindex(B, 2)
            for i in firstindex(B, 1):(j - 1)
                y[i] += B[i, j] * x[j]
            end
        end
        @inbounds for j in axes(B, 2)
            for i in (j + 1):lastindex(B, 2)
                y[i] += B[i, j] * y[j]
            end
        end
        @. y = (1 - ω) * x + ω * y
    end
end

end # module
