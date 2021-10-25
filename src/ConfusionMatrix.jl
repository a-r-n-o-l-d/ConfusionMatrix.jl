module ConfusionMatrix

using NamedArrays
#using NamedArrays: NamedMatrix
using Flux: OneHotArray, OneHotMatrix, onecold
import Base.push!
import Base.show

export ConfMat, update!, metrics, metrics!, toidx

include("confmat.jl")
include("metrics.jl")

#=export ConfMatrix

using NamedArrays
using Flux: OneHotMatrix, onecold

struct ConfMatrix{N}
    labs
    cmat
end

ConfMatrix(l, m) = ConfMatrix{length(l)}(l, m)

function ConfMatrix(l)
    n = length(l)
    m = NamedArray(zeros(Int, n, n), (l, l), ("Actual", "Predicted"))
    ConfMatrix(l, m)
end

function ConfMatrix(l, x, y)
    cm = ConfMatrix(l)
    push!(cm, x, y)
end

function ConfMatrix(l, x, y; thrd)
    cm = ConfMatrix(l)
    push!(cm, x, y, thrd = thrd)
end

Base.show(io::IO, cm::ConfMatrix) = show(cm.cmat)

function labidx(v, l)
    i = indexin(v, l)
    all(ismissing.(i)) && throw(ArgumentError(
        "Unknown labels in supplied data."))
    Vector{Int}(i)
end

function Base.push!(cm::ConfMatrix, x, y::Vector{T}; thrd) where T <: AbstractFloat
    x = labidx(x, cm.labs)
    y = @. Int(y > thrd) + 1
    push!(cm, x, y)
end

function Base.push!(cm::ConfMatrix, x::T, y::T) where T <: OneHotMatrix
    x = onecold(x)
    y = onecold(y)
    push!(cm, x, y)
end

function Base.push!(cm::ConfMatrix, x::Vector{T}, y::Vector{T}) where T
    x = labidx(x, cm.labs)
    y = labidx(y, cm.labs)
    push!(cm, x, y)
end

function Base.push!(cm::ConfMatrix, x::Vector{Int}, y::Vector{Int})
    length(y) == length(x) || throw(DimensionMismatch(
        "Number of actual values and number of predicted values mismatch."))
    for i in eachindex(y)
        @inbounds r = x[i]
        @inbounds c = y[i]
        cm.cmat[r, c] += 1
    end
    cm
end
=#
end