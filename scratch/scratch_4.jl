using Base




mutable struct Test
    v
end

Base.show(io::IO, x::Test) = print(io, "Test($(x.v))")


function inverted(x::Test)
    x.v = 1 / x.v
end

function tmp(x)
    println(x)
end

x = Test(10.0)

println(x)

inverted(x) do x
    println(x)
end

println(x)