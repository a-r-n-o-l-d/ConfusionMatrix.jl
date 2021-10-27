using ConfusionMatrix
using Test
using Flux: onehotbatch, softmax

@testset "ConfusionMatrix.jl" begin
    @testset "ConfMat construction" begin

# ConfMat([:truc, :bidule], [:truc, :truc, :truc, :bidule], [:truc :truc :bidule :bidule])
# ConfMat([:truc, :bidule, :machin], [:truc, :truc, :truc, :bidule], [:truc :truc :bidule :machin])
# ConfMat([0 1], [0, 1, 1, 0], [0, 1, 1, 0])

        labels = [:dog, :cat]
        y = rand(labels, 100)
        ŷ = rand(labels, 100)
        cm = ConfMat(labels, y, ŷ)
        @test size(cm.counts) == (2, 2)

        labels = [:dog, :cat, :frog]
        y = rand(labels, 100)
        ŷ = rand(labels, 100)
        cm = ConfMat(labels, y, ŷ)
        @test size(cm.counts) == (3, 3)

        labels = [1, 2, 3, 4, 5]
        y = rand(labels, 100)
        ŷ = rand(labels, 100)
        cm = ConfMat(labels, y, ŷ)
        @test size(cm.counts) == (5, 5)

        # Simulate the ouput of a Flux binary classifier
        labels = [:dog, :cat]
        cm = ConfMat(labels)
        nepochs = 10
        batchsize = 64
        for e ∈ 1:nepochs
            y = onehotbatch(rand(labels, batchsize), labels)
            ŷ = rand(1, batchsize)
            update!(cm, y, ŷ)
        end
        @test sum(cm.counts) == nepochs * batchsize

        # Simulate the ouput of a Flux multiclass classifier
        labels = [:dog, :cat, :frog, :bear]
        cm = ConfMat(labels)
        nepochs = 10
        batchsize = 64
        for e ∈ 1:nepochs
            y = onehotbatch(rand(labels, batchsize), labels)
            ŷ = rand(4, batchsize) |> softmax
            update!(cm, y, ŷ)
        end
        @test sum(cm.counts) == nepochs * batchsize

        io = IOBuffer()
        show(io, cm)
        @test startswith(String(take!(io)), "ConfMat{4,Symbol}")
    end
end
