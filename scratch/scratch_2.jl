# macro build(type)
#     quote
#         function $(esc(type))(params::T) where T
#             $(Expr(:call, esc(:(new)), :params))
#         end
#     end
# end

# struct Test_2
#     x
#     @build(Test_2)
# end

# Test_2(10)


# macro build(type)
#     quote
#         function $(esc(type))(params::T) where T
#             n = Expr(:escape, :(new))
#             $(Expr(:call, esc(:(new)), :params))
#         end
#     end
# end



# macro build(type, T)
#     quote
#         function $(esc(type))(params::$T)
#             $(Expr(:call, Expr(:escape, :(new{$T})), :params))
#         end
#     end
# end

# struct Test_3{T}
#     x::T
#     @build(Test_3, Int64)
# end

# Test_3(10)



# macro build_2(type)
#     quote
#         function $(esc(type))(params::T) where T
            
#             $(Expr(:call, Expr(:escape, :(new)), :params))
#         end
#     end
# end


# macro build(type, T)
#     quote
#         function $(esc(type))(params::$T)
#             $(Expr(:call, Expr(:escape, :(new{$T})), :params))
#         end
#     end
# end

# struct Test_3{T}
#     x::T
#     @build(Test_3, Int64)
# end

# Test_3(10)



# struct Test_2
#     x
#     @build(Test_2)
# end

# Test_2(10)

# struct Test_3{T}
#     x::T
#     @build(Test_3)
# end

# Test_3(10)



# struct Test{T}
#     x::T
#     @build(Test::T) where T
# end


# macro build(type, N)
#     quote
#         function $(esc(type))(v::Dict{K, V}) where {K<:NTuple{$N, I}, V} where I<:Integer
#             $(Expr(:call, Expr(:escape, :(new{K, V})), :v))
#         end
#     end
# end

macro build(type)
    quote
        function $(esc(type))(v::T) where T<:Integer
            v = $(:(new{Int64}))
            $(Expr(:call, Expr(:escape, $v), :params))
        end
    end
end

macro build(type)
    quote
        function $(esc(type))(v::T) where T<:Integer
            $(Expr(:call, esc(:(new{Int64})), :v))
        end
    end
end

macro build(type)
    quote
        function $(esc(type))(v::T) where T<:Integer
            $(Expr(:call, esc(:(new{T})), :v))
        end
    end
end

struct Test{T}
    x::T
    @build(Test)
end

Test(10)

macro build_2(type)
    quote
        function $(esc(type))(params::T) where T<:Integer
            $(Expr(:call, Expr(:escape, :(new{Int64})), :params))
        end
    end
end





# macro build(type, N)
#     quote
#         function $(esc(type))(v::Dict{K, V}) where {K<:NTuple{$N, I}, V} where I<:Integer
#             $(Expr(:call, Expr(:escape, :(new{NTuple{$N, :I}, Int64})), :v))
#         end
#     end
# end

# struct AtomicParams_2{K, V}
#     vals::Dict{K, V}

#     @build(AtomicParams_2, 1)
#     @build(AtomicParams_2, 2)
# end

# AtomicParams_2(Dict((1,2)=>2))



# struct AtomicParams_1{K, V}
#     vals::Dict{K, V}
#     AtomicParams_1(v::Dict{K, V}) where {K<:NTuple{1, I}, V} where I<:Integer = new{K, V}(v)
#     AtomicParams_1(v::Dict{K, V}) where {K<:NTuple{2, I}, V} where I<:Integer = new{K, V}(v)
# end


# struct AtomicParams_2{K, V}
#     vals::Dict{K, V}

#     @build(AtomicParams_2, 1)
#     @build(AtomicParams_2, 2)
# end

