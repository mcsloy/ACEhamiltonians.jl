using ACE: SphericalMatrix
using SparseArrays
using ACEbase.ObjectPools: VectorPool
# This contains information about combining duel models into a single object.

# Convert ℓᵢℓⱼ specific entities/types into their ℓⱼℓᵢ counterparts

# SphericalMatrix instance 
_invert(m::T) where T<:SphericalMatrix = SphericalMatrix{T.parameters[[2,1,4,3,5,6]]...}(m.val[end:-1:1,end:-1:1]')

# SphericalMatrix type
_invert(::Type{T}) where T<:SphericalMatrix = SphericalMatrix{T.parameters[[2,1,4,3,5,6]]...}

# B_pool instances
_invert(::VectorPool{T₁}) where {T₁<:SphericalMatrix} = VectorPool{_invert(T₁)}()

# A2Bmap instances; multiple dispatch here halves the execution time 
function _invert(A2Bmap::SparseMatrixCSC{T₁, I}, ::Type{T₂}) where {T₁<:SphericalMatrix, I<:Integer, T₂<:SphericalMatrix}
    return SparseMatrixCSC{T₂, I}(
        A2Bmap.m, A2Bmap.n, A2Bmap.colptr, A2Bmap.rowval,
        T₂[T₂(i.val[end:-1:1,end:-1:1]') for i in A2Bmap.nzval])
end

function _invert(A2Bmap::SparseMatrixCSC{T₁, I}) where {T₁<:SphericalMatrix, I<:Integer}
    return _invert(A2Bmap, _invert(T₁))
end

invert_s(basis::T) where T<:SymmetricBasis = SymmetricBasis(
    basis.pibasis, _invert(basis.A2Bmap), basis.symgrp, basis.real, _invert(basis.B_pool))


function acquire_B!(basis::Basis, invert::Bool=False)
    VT = ACE.valtype(basis, args...)
    if hasproperty(basis, :B_pool)
       return acquire!(basis.B_pool, length(basis), VT)
    end 
    return Vector{VT}(undef, length(basis))
 end

evaluate(basis::ACEBasis, args...) = evaluate!( acquire_B!(basis, args...), basis, args... )


# function evaluate!(B, C, basis::SymmetricBasis, cfg::AbstractConfiguration)
#     AA = acquire_B!(basis.pibasis, cfg)
#     evaluate!(AA, basis.pibasis, cfg)
#     return genmul!(B, C, AA, (a, b) -> basis.real(a * b))
# end

# function evaluate!(basis::T, cfg::AbstractConfiguration) T<:Basis
#     evaluate!(B₁, C₁, basis.basis, cfg)
#     evaluate!(B₂, C₂, basis.basis, cfg)
#     acquire_B!(basis, args...)
# end

function evaluate!(basis::T, states::AbstractConfiguration)
    A = evaluate!(basis.basis, ACEConfig(states))
end

function evaluate!(basis::T, states::S) where {T<:Basis, S<:AbstractState}
    A = evaluate!(basis.basis, ACEConfig(states))
    B = evaluate_real_new(A)
    values .= (basis.coefficients' * B) + basis.mean
    if !ison(basis)
        A = evaluate!(_invert(basis.basis), ACEConfig(reverse.(states)))
        B = evaluate_real_new(A)

    end

    A = evaluate!(_invert(basis.basis), ACEConfig(reverse.(states)))
end











abstract type Basis{T} where T<:AbstractFloat end



struct OnSiteBasis{T} <: Basis{T}
    basis
    id
    coefficients
end

struct OffSiteBasis{T} <: Basis{T}
    basis
    id
    coefficients_i
    coefficients_j
    mean_i
    mean_j
end
    

# end