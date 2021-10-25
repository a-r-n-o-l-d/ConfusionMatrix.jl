"""

labels = [:dog, :cat];
y = rand(labels, 100);
ŷ = rand(labels, 100);
ConfMat(labels, y, ŷ)


labels = [:dog, :cat, :frog];
y = rand(labels, 100);
ŷ = rand(labels, 100);
ConfMat(labels, y, ŷ)

labels = [:dog, :cat, :frog];
y = rand(1:3, 100);
ŷ = rand(1:3, 100);
ConfMat(labels, y, ŷ)

labels = [:dog, :cat];
y = [:dog, :cat, :dog, :dog];
ŷ = [0.1, 0.8, 0.7, 0.1];
ConfMat(labels, y, ŷ, threshold = 0.5)

labels = [:dog, :cat, :frog];
y = [:dog, :frog, :dog, :dog];
ŷ = [:dog, :frog, :cat, :dog];
ConfMat(labels, y, ŷ)

labels = [2, 3, 1];
y = [1, 1, 3, 2];
ŷ = [2, 2, 2, 3];
ConfMat(labels, y, ŷ)

labels = [:dog, :cat, :frog];
y = onehotbatch([:dog, :frog, :dog, :dog], labels);
ŷ = [0.8 0.2 0.3 0.5; 0.1 0.1 0.9 0.2; 0.1 0.1 0.1 0.1];
ConfMat(labels, y, ŷ)


labels = [:dog, :cat];
y = onehotbatch([:dog, :cat, :dog, :cat], labels)
ŷ = [0.8 0.1 0.2 0.8]
ConfMat(labels, y, ŷ)




labels = [:dog, :cat];
y = onehotbatch([:dog, :cat, :dog, :cat], labels)
ŷ = [0.8 0.1 0.2 0.8]
cm = ConfMat(labels)
update!(cm, y, ŷ)
cm.counts
update!(cm, y, ŷ, 0.5)
cm.counts

labels = [:dog, :cat, :frog];
y = onehotbatch([:dog, :frog, :dog, :dog], labels);
ŷ = [0.8 0.2 0.3 0.5; 0.1 0.1 0.9 0.2; 0.1 0.1 0.1 0.1];
cm = ConfMat(labels)
update!(cm, y, ŷ, 0.5)
"""
struct ConfMat{N, T}
    labels::Vector
    counts::NamedMatrix
    # threshold::AbstractFloat
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

function update!(cm::ConfMat{2,}, y::OneHotMatrix, ŷ::AbstractMatrix)
    y = onecold(y, cm.labels)
    ŷ = onecold(ŷ, cm.labels)
    update!(cm, y, ŷ)
end

function update!(cm::ConfMat{2,}, y::OneHotMatrix, ŷ::AbstractMatrix, threshold)
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

# function ConfMat(labels::Vector, y::T, ŷ::T; threshold = 0.5) where {T <: Vector{Int}}
#     cm = ConfMat(labels, threshold = threshold)
#     update!(cm, y, ŷ)
# end

# function ConfMat(labels, y, ŷ; threshold = 0.5)
#     labels = vec(labels)
#     cm = ConfMat(labels, threshold = threshold)
#     update!(cm, y, ŷ)
# end

# function toidx(cm::ConfMat{N, T}, x::Vector{T}) where {N, T}
#     labidx(x, cm.labels)
# end

# function toidx(cm::ConfMat{N, T}, x::AbstractMatrix) where {N, T}
#     x = onecold(x, cm.labels)
#     labidx(x, cm.labels)
# end

# function toidx(cm::ConfMat{2,}, x::OneHotMatrix)
#     x = onecold(x, cm.labels)
#     labidx(x, cm.labels)
# end

# function toidx(cm::ConfMat{2,}, y::OneHotMatrix, ŷ::AbstractMatrix)
#     y = onecold(y, cm.labels)
#     labidx(y, cm.labels), @. Int(ŷ > cm.threshold) + 1
# end

# function toidx(cm::ConfMat{2,}, x::Vector{T}) where T <: AbstractFloat
#     x = @. Int(x > cm.threshold) + 1
# end

# function toidx(cm::ConfMat{2,}, x::Matrix{T}) where T <: AbstractFloat #where {E <: AbstractFloat, T <: Union{Matrix{E}, Vector{E}}}
#     x = vec(x)
#     x = @. Int(x > cm.threshold) + 1
# end

# function toidx(cm::ConfMat{2,}, x::AbstractArray{T}) where T <: AbstractFloat
#     x = vec(x)
#     x = @. Int(x > cm.threshold) + 1
# end

# function update!(cm::ConfMat, y, ŷ)
#     y = toidx(cm, y)
#     ŷ = toidx(cm, ŷ)
#     update!(cm, y, ŷ)
# end

function update!(cm::ConfMat{N, T}, a::Vector{T}, p::Vector{T}) where {N, T}
    a = labidx(a, cm.labels)
    p = labidx(p, cm.labels)
    _update!(cm, a, p)
end

# Probleme si label Int update! => _update!
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

# function toidx(x::OneHotMatrix, labels::Vector)
#     x = onecold(x, labels)
#     labidx(x, labels)
# end

# function toidx(x::AbstractMatrix, labels::Vector)
#     x = onecold(x, labels)
#     labidx(x, labels)
# end

# function toidx(x::AbstractMatrix, threshold::AbstractFloat) #, labels
#     x = @. Int(x[:] > threshold) + 1
# end

# function toidx(cm::ConfMat{N,}, x::T) where {N, T <: Union{AbstractMatrix, OneHotMatrix}}
#     x = onecold(x, labels)
#     labidx(x, labels)
# end

# function ConfMat(labels::Vector, y::OneHotMatrix, ŷ::AbstractMatrix)
#     cm = ConfMat(labels)
#     update!(cm, y, ŷ)
# end

# function ConfMat(labels::Vector, y::OneHotMatrix, ŷ::AbstractMatrix; threshold)
#     cm = ConfMat(labels)
#     println(y)
#     update!(cm, y, ŷ; threshold = threshold)
# end

# function ConfMat(labels::Vector{T1}, a::Vector{T1}, p::Vector{T2};
#                  threshold) where {T1, T2 <: AbstractFloat}
#     cm = ConfMat(labels)
#     update!(cm, a, p; threshold = threshold)
# end

# function show(io::IO, cm::ConfMat)
#     println(io, "Confusion matrix")
#     print(io, "labels = ")
#     show(io, cm.labels)
#     println(io)
#     #println(io, "contingency table")
#     show(io, cm.counts)
# end

# function update!(cm::ConfMat{2, T1}, a::Vector{T1}, p::Vector{T2};
#                  threshold) where {T1, T2 <: AbstractFloat}
#     a = @view a[:]
#     p = @view p[:]
#     a = labidx(a, cm.labels)
#     p = @. Int(p > threshold) + 1
#     update!(cm, a, p)
# end

# function update!(cm::ConfMat{2, T}, y::OneHotArray, ŷ::AbstractMatrix; threshold) where T
#     println(y)
#     y = onecold(y, cm.labels)
#     y = labidx(y, cm.labels)
#     ŷ = @. Int(ŷ[:] > threshold) + 1
#     update!(cm, y, ŷ)
# end

# function update!(cm::ConfMat, y::OneHotMatrix, ŷ::AbstractMatrix)
#     y = onecold(y, cm.labels)
#     ŷ = onecold(ŷ, cm.labels)
#     update!(cm, y, ŷ)
# end

# function update!(cm::ConfMat{2,}, y::OneHotMatrix, ŷ::AbstractMatrix; threshold)
#     y = onecold(y, cm.labels)
#     y = labidx(y, cm.labels)
#     ŷ = @view ŷ[:] #onecold(ŷ, cm.labels)
#     ŷ = @. Int(ŷ > threshold) + 1
#     update!(cm, y, ŷ)
# end