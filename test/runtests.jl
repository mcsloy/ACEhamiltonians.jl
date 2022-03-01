using ACEhamiltonians
using Test

@testset "ACEhamiltonians.jl" begin
    include("matrix_tolerance_tests.jl")
end

@testset "ParaDef.jl" begin
    include("unit_tests/test_paradef.jl")
end
