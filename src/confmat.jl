"""
    ConfMat(labels)
    ConfMat(labels, y, ŷ)

# Example
```julia
julia> labels = [:dog, :cat]
2-element Vector{Symbol}:
 :dog
 :cat

 julia> y = rand(labels, 100);

 julia> ŷ = rand(labels, 100);
 
 julia> cm = ConfMat(labels, y, ŷ)
 ConfMat{2, Symbol}([:dog, :cat], 2×2 Named Matrix{Int64}
 Actual class ╲ Predicted class │ :dog  :cat
 ───────────────────────────────┼───────────
 :dog                           │   24    26
 :cat                           │   24    26)

```
"""
struct ConfMat{N, T}
    labels
    counts
end

function ConfMat(labels::Vector, counts::NamedMatrix)
    N = length(labels)
    T = eltype(labels)
    ConfMat{N, T}(labels, counts)
end

function ConfMat(labels::Vector)
    #labels = vec(labels)
    n = length(labels)
    c = zeros(Int, n, n)
    c = NamedArray(c, (labels, labels), ("Actual class", "Predicted class"))
    ConfMat(labels, c)
end

function ConfMat(labels::T, y::T, ŷ::T) where {E, T <: Vector{E}}
    cm = ConfMat(labels)
    update!(cm, y, ŷ)
end

function update!(cm::ConfMat{2,}, y::OneHotMatrix, ŷ::AbstractMatrix;
                 threshold = 0.5)
    y = onecold(y, cm.labels)
    y = labidx(y, cm.labels)
    ŷ = (@. Int(ŷ > threshold) + 1) |> vec
    _update!(cm, y, ŷ)
end

function update!(cm::ConfMat{N,}, y::OneHotMatrix, ŷ::AbstractMatrix) where N
    y = onecold(y, cm.labels)
    ŷ = onecold(ŷ, cm.labels)
    update!(cm, y, ŷ)
end

function update!(cm::ConfMat{N, T}, a::Vector{T}, p::Vector{T}) where {N, T}
    a = labidx(a, cm.labels)
    p = labidx(p, cm.labels)
    _update!(cm, a, p)
end

function _update!(cm::ConfMat, y::Vector{Int}, ŷ::Vector{Int})
    length(ŷ) == length(y) || throw(DimensionMismatch(
        "Number of actual values and number of predicted values mismatch."))
    for i in eachindex(ŷ)
        @inbounds r = y[i]
        @inbounds c = ŷ[i]
        @inbounds cm.counts[r, c] += 1
    end
    cm
end

"""
    labidx(v, l)

"""
function labidx(v, l)
    i = indexin(v, l)
    all(.!isnothing.(i)) || throw(ArgumentError(
        "Unknown labels in supplied data."))
    Vector{Int}(i)
end

summary(cm::ConfMat{N, T}) where {N, T} = 
    string("Confusion matrix of ", T, " with ", N, " labels ")

Base.print(cm::ConfMat) = print(cm.counts)

function Base.show(io::IO, cm::ConfMat)
    print(io, summary(cm))
    show(io, cm.labels)
    tmp = IOBuffer()
    show(tmp, cm.counts)
    println(io)
    sh = String(take!(tmp))
    for l ∈ split(sh, "\n")[2:end]
        println(io, l)
    end
end