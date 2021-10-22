using ConfusionMatrix
using Test
using Flux: onehotbatch

@testset "ConfusionMatrix.jl" begin
    @testset "ConfMat construction" begin
        l = [0 1]
        a = [0 1 1 0]
        p = [0.1 0.8 0.2 0.9]
        cm = ConfMat(l, a, p, threshold = 0.5)
        @test cm.cmat[1, 1] == 1
        @test cm.cmat[2, 2] == 1

        l = [0, 1]
        a = [0, 1, 1, 0]
        p = [0.1, 0.8, 0.2, 0.9]
        cm = ConfMat(l, a, p, threshold = 0.5)
        @test cm.cmat[1, 1] == 1
        @test cm.cmat[2, 2] == 1

        l = [:truc, :bidule, :machin]
        a = onehotbatch([:truc, :machin, :truc, :machin], l)
        p = [0.8 0.2 0.3 0.5; 0.1 0.1 0.9 0.2; 0.1 0.1 0.1 0.1]
        cm = ConfMat(l, a, p)
        @test cm.cmat[3, 1] == 2

        p = onehotbatch([:truc, :truc, :machin, :bidule, :bidule], l)
        @test_throws DimensionMismatch ConfMat(l, a, p)

        a = [:truc, :machin, :truc, :machin]
        p = [:truc, :truc, :machin, :chouette]
        @test_throws ArgumentError ConfMat(l, a, p)

        a = [:truc, :machin, :machin, :bidule]
        p = [:truc, :truc, :machin, :bidule]
        cm = ConfMat(l, a, p)
        @test cm.cmat[1, 1] == 1
        @test cm.cmat[2, 2] == 1
        @test cm.cmat[3, 3] == 1


        l = [1, 2, 3]
        a = [3, 1, 2, 3]
        p = [3, 2, 2, 3]
        cm = ConfMat(l, a, p)
        @test cm.cmat[1, 1] == 0
        @test cm.cmat[2, 2] == 1
        @test cm.cmat[3, 3] == 2
    end
end
