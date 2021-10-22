struct ConfMat{N, T}
    labs
    cmat
end

ConfMat(l, m) = ConfMat{length(l), eltype(l)}(l, m)

function ConfMat(l)
    l = vec(l)
    n = length(l)
    m = NamedArray(zeros(Int, n, n), (l, l), ("Actual", "Predicted"))
    ConfMat(l, m)
end

function ConfMat(l, a, p)
    cm = ConfMat(l)
    push!(cm, a, p)
end

function ConfMat(l, a, p; threshold::AbstractFloat)
    cm = ConfMat(l)
    push!(cm, a, p, threshold = threshold)
end

show(io::IO, cm::ConfMat) = show(cm.cmat)

function push!(cm::ConfMat{2, T1},
               a::AbstractArray{T1},
               p::AbstractArray{T2}; threshold) where {T1, T2 <: AbstractFloat}
    a = @view a[:]
    p = @view p[:]
    a = labidx(a, cm.labs)
    p = @. Int(p > threshold) + 1
    push!(cm, a, p)
end

function push!(cm::ConfMat, a::OneHotMatrix, p::AbstractMatrix)
    a = onecold(a, cm.labs)
    p = onecold(p, cm.labs)
    push!(cm, a, p)
end

function push!(cm::ConfMat{N, T}, a::Vector{T}, p::Vector{T}) where {N, T}
    a = labidx(a, cm.labs)
    p = labidx(p, cm.labs)
    push!(cm, a, p)
end

function push!(cm::ConfMat, a::Vector{Int}, p::Vector{Int})
    length(p) == length(a) || throw(DimensionMismatch(
        "Number of actual values and number of predicted values mismatch."))
    for i in eachindex(p)
        @inbounds r = a[i]
        @inbounds c = p[i]
        @inbounds cm.cmat[r, c] += 1
    end
    cm
end

function labidx(v, l)
    i = indexin(v, l)
    all(.!isnothing.(i)) || throw(ArgumentError(
        "Unknown labels in supplied data."))
    Vector{Int}(i)
end