module DataManipulation
using BlockArrays, LinearAlgebra, StaticArrays, HDF5, NeighbourLists
using HDF5: Group
using ACE: BondEnvelope, State, PositionState, AbstractState, CylindricalBondEnvelope
using ACEbase: AbstractConfiguration
using Base
using JuLIP
using JuLIP: Atoms
using JuLIP.Utils: project_min
using ACEhamiltonians
using ACEhamiltonians.Common: BasisDef


export collect_data, collect_data_from_hdf5, DataSet, locate_blocks, upper_blocks, off_site_blocks, AbstractData, filter_sparse, filter_bond

# Bond envelopes should be forced to have their λ fields renamed an set to zero, then
# the state positions could be made relative to zero.
 

# Todo:
#   - `be` should be replaced by a boolean so that state size is consistent. This will
#     speed up state construction.
#   - An ACEConfig like entity should be created for batches of states.

# Tis should replace the BondState entity.
# struct BondState{T₁<:SMatrix{<:Integer, 3, <:AbstractFloat}, T₂<:SVector{3, <:AbstractFloat}, S<:Symbol}
#     rr::T₁
#     rr0::T₂
#     be::S
# end




# These structures are used rather than the standard tuple based ACE state as they
# are much faster (take 80% less time)
struct BondState{T<:SVector{3, <:AbstractFloat}, S<:Symbol} <: AbstractState
    rr::T
    rr0::T
    be::S
end

struct AtomState{T<:SVector{3, <:AbstractFloat}} <: AbstractState
    rr::T
end

# struct Config{T<:Vector{<:AbstractState}} <: AbstractConfiguration
#     states
# end

function Base.show(io::IO, state::BondState)
    rr = string([round.(state.rr, digits=5)...])
    rr0 = string([round.(state.rr0, digits=5)...])
    print(io, "BondState(rr:$rr, rr0:$rr0, env:$(state.be))")
end

function Base.show(io::IO, state::AtomState)
    rr = string([round.(state.rr, digits=5)...])
    print(io, "AtomState(rr:$rr)")
end

Base.:(==)(x::T, y::T) where T<:BondState = x.rr ≡ y.rr && y.rr0 ≡ y.rr0 && x.be ≡ y.be
Base.:(==)(x::T, y::T) where T<:AtomState = x.rr ≡ y.rr

Base.isapprox(x::T, y::T; kwargs...) where T<:BondState = isapprox(x.rr, y.rr; kwargs...) && isapprox(x.rr0, y.rr0; kwargs...) && x.be ≡ y.be
Base.isapprox(x::T, y::T; kwargs...) where T<:AtomState = isapprox(x.rr, y.rr; kwargs...)

Base.isapprox(x::T, y::T; kwargs...) where T<:AbstractVector{<:BondState} = all(x .≈ y)
Base.isapprox(x::T, y::T; kwargs...) where T<:AbstractVector{<:AtomState} = all(x .≈ y)

Base.zero(::Type{BondState{T, S}}) where {T, S} = BondState{T, S}(zero(T), zero(T), :O)
Base.zero(::Type{AtomState{T}}) where T = AtomState{T}(zero(T))
Base.zero(::B) where B<:BondState = zero(B)
Base.zero(::B) where B<:AtomState = zero(B)


Parameters.ison(::T) where T<:AtomState = true
Parameters.ison(::T) where T<:BondState = false

# `Base.reverse` should be removed in favour of `Base.inv`

"""

Reverse a `BondState` so that the state is from the perspective of the second atom.
It is important to note that this assumes that the bond origin lies between the two
bonding atoms. 
"""
function Base.reverse(state::BondState{T, S}) where {T, S}
    if state.be == :bond
        return BondState{T, S}(-state.rr, -state.rr0, state.be)
    else
        return BondState{T, S}(state.rr, -state.rr0, state.be)
    end
end

function Base.inv(state::BondState{T, S}) where {T, S}
    if state.be == :bond
        return BondState{T, S}(-state.rr, -state.rr0, state.be)
    else
        return BondState{T, S}(state.rr, -state.rr0, state.be)
    end
end

# Todo:
#   - Due to the caching effects of the neighbour lists the `get_states` methods
#   - will commonly include species beyond the expected range.  
#   - Rename `get_states` to `get_state` and implement an auto-vectorised version.
#   - This code should be partitioned into separate, more relevant, modules. Such as an
#     io module and a data manipulation module. If possible much of this should be
#     generalised and abstracted to a separate ACE package.
#   - Data manipulation subroutines should be more clearly named and stratified into
#     various levels, i.e. manually collecting and passing matrices vs passing a database
#     and having the code autoload the relevant data.  
#   - When the geometry is stored there should be:
#      - an attribute flag indicating if the system is molecular.
#      - a pbc flag which indicates the periodicity of non-molecular systems.
#      - a hash code that uniquely identifies the system.
#   - Units should be checked when loading an Atoms object from HDF5.
#   - Add support for cutoff specification.


# There are some issues with how data is currently gathered. Structural information is
# filtered to generate the environmental data, just for this data to be filtered again
# by the model.

# Need a method by which to specify the target blocks
# The base method will be manual selection of atom pairs
# There will be higher level constructors which allow for a more curated selection process



RefAtoms = Base.RefValue{Atoms{I}} where I

"""
Tasks:

    - Add multi-species support to the on/off-site state constructor methods.
    - Create dereferencing macro that allows for references to be provided for atoms
      etc. This helps with vectorisation. The result should look something like:
      get_states(i, atoms::RefAtoms; args...) = get_states(i, atoms[]; args...) 
"""


"""
    get_states(i, j, atoms[; r])

Construct a sate vector describing the local environment about a target atom.

# Arguments
- `i::Integer`: atomic index of the target atom.
- `atoms::Atoms`: the atomic system to which `i` relates.
- `r::AbstractFloat`: maximum view distance for the environment, defaults to 20 Å.

# Returns
- `states::Vector{State}`: a vector of environment `State` entities specifying the
   relative positions of all environmental atoms.

# Todo
- Add multi-species support by adding species information to the state.

# Notes
Cell translation vectors are not required for on-site state construction as 1) they have
no effect on the local environment, and ii) on-sites only exist in the origin cell.

# Warnings
The environment distance cutoff `r` acts only as a minimum distance. As such is it
possible that species beyond this distances may be returned. This is because the
neighbour list is cached by the `Atoms` object to improve performance.
"""
function get_states(i::I, atoms::Atoms; r::F=20.0) where {I<:Integer, F<:AbstractFloat}
    # Construct the neighbour list (this is cached so speed is not an issue)
    pair_list = JuLIP.neighbourlist(atoms, r; fixcell=false)

    # Extract environment about each relevant atom from the pair list. These will be tuples
    # of the form: (atomic-index, relative-position)
    environments = NeighbourLists.neigs(pair_list, i)

    # zs = getindex.(Ref(atoms.Z), map(i->i[1], environments))  # ← to be added later
    # return PositionState.(environments[2])
    return AtomState.(environments[2])
end


"""
    get_states(i, j, atoms[; image, r])

Construct a sate vector describing the local environment about a target bond.

# Arguments
- `i::Integer`: atomic index of the first bond atom.
- `j::Integer`: atomic index of the second bond atom.
- `atoms::Atoms`: the atomic system to which the bond relates.
- `envelope::BondEnvelope`: bond envelope used to refine local environment.
- `image::Vector{Integer}`: the periodic cell image in which atom `j` resides. If no value
    is provided then j will be assumed to resided in the closest periodic image.

# Returns
- `states::Vector{State}`: vector of environment `State`s specifying the relative
   positions of environmental atoms & the bond vector. Note, the firs state is a bond
   bond state indicating bond vector and relative positions of the bond atoms.

# Todo
- Add multi-species support by adding species information to the state.
- Resolve issue caused by considering only the environment around atom i.
- bond origin should be set the the midpoint between the two atoms rather than the
  position of the first atom.
- Environmental culling should either i) not be performed here or ii) be performed in
  the on-site state constructor too.

This should be partitioned into two different methods, on which takes images and another
that does not. Furthermore, the `NeighbourLists.neigs` function should be replaced with the
`NeighbourLists.neigss` function and the information returned used to bypass manually
computing rr0. The neigss method should be replaced by a method which returns a static
array rather than a set of static vectors.

For some reason the JuLIP.neighbourlist call takes a long time even if only retrieving the
cache

# Warnings
The environment distance cutoff `r` acts only as a minimum distance. As such is it
possible that species beyond this distances may be returned. This is because the
neighbour list is cached by the `Atoms` object to improve performance.
"""
function get_states(i::I, j::I, atoms::Atoms, envelope::CylindricalBondEnvelope, image::Union{AbstractArray{I}, Nothing}=nothing) where {I<:Integer}

    # Warning this assumes that positions are correctly wrapped in the cell.

    # Check whether or not we want the bond rr0 to be from atom 1 to 2 or from
    # from atom 1 to the mid-point.


    # Todo:
    #   - If the distance between atoms i and j is greater than the cutoff distance r
    #     then it is likely that an error will be encountered. A safety catch should be
    #     built in to handle this scenario.
    
    # Identify an appropriate cutoff, this must take into account the fact that distances will
    # be calculated with respect to the first atom, but we will need them to be relative to the
    # centre of the bond. 
    r = sqrt((envelope.r0cut + envelope.zcut)^2 + envelope.rcut^2)
    # Get the indices & distance vectors of atoms neighbouring i
    idxs, vecs, cells = neighbours(i, atoms, r)

    # Get the bond vector between atoms i & j; where i is in the origin cell & j resides
    # in either i) closest periodic image, or ii) that specified by `image` if provided.
    if isnothing(image)
        idx = locate_minimum_image(j, idxs, vecs)
        rr0 = vecs[idx]
    else
        idx = locate_target_image(j, idxs, cells, image)
        if idx == 0
            x = atoms.X[j] - atoms.X[i]
            rr0 = x + (adjoint(image .* atoms.pbc) * atoms.cell).parent
        else
            rr0 = vecs[idx]
        end
    end

    # Offset needed to shift positions so that they are relative to the bond origin. 
    offset = envelope.λ == 0 ? rr0 / 2 : zero(rr0)

    # Bond state; "rr"=position of i relative j, "rr0"=bond vector & "be" indicates
    # that this is a bond (rr==rr0 here).
    bond_state = BondState(rr0, rr0, :bond)

    # Environmental atoms states, "rr0"=bond=vector, "rr"=atom's position relative to i &
    # "be" indicates that this is an environmental state.
    t = typeof(rr0)
    env_states = BondState{t, Symbol}[]
    sizehint!(env_states, length(vecs))
    for vec in vecs[1:end .!= idx]
        # If applicable, offset the vectors so the bond origin lies at the midpoint
        # between the two atoms.
        vec = vec - offset

        # If the an environmental atom lies too close to the origin it must be offset
        # to avoid errors. While this introduces noise, it is better than not being
        # able to fit. A better solution should be found where and if possible. 
        vec_norm = norm(vec)
            
        if vec_norm <= 0.05
            @warn "Some environmental atoms lie too close to the bond origin and have been offset!" maxlog=1
            if vec_norm == 0
                # If the atom is simply too close, then move it further along its bond vector
                vec = t(normalize(rand(3)) * 0.08)
            else
                # If the atom lies exactly at the bond origin, then offset it along the bond
                # vector in the direction of the atom with the lowest atomic index, or if both
                # are the same then atom i. This ensures the offset is consistent and reproducible.
                o = i <= j ? -1 : 1
                vec = normalize(rr0) * 0.08 * o
            end
        end
        push!(env_states, BondState{t, Symbol}(vec, rr0, :env))

    end

    # env_states = BondState{typeof(rr0), Symbol}[BondState(v - offset, rr0, :env) for v in vecs[1:end .!= idx]]
    
    # Cull states outside of the bond envelope. A double filter is required as the
    # inbuilt filter operation deviates from standard julia behaviour.
    env_states = Base.filter(x -> filter(envelope, x), env_states)

    return [bond_state; env_states]

end


function neighbours(i::I, atoms::Atoms, r::F) where {I<:Integer, F<:AbstractFloat}
    pair_list = JuLIP.neighbourlist(atoms, r; fixcell=false)
    return NeighbourLists.neigss(pair_list, i)
end

"""
# Arguments
- `j<:Integer`: Index of the atom for for whom the minimum image is to be identified.
- `idxs::Vector{<:Integer}`: Integers specifying the indices of the atoms to two which
  the distances in `vecs` correspond.
- `vecs::Vector{SVector{3, <:AbstractFloat}}`: Vectors between the the source atom and
  the target atom.


# Notes
Note that this will return the index in `i` for which `vecs[i]` will return the shortest
vector between the source atom and atom number `j`.
"""
function locate_minimum_image(j::I, idxs::AbstractVector{I}, vecs::AbstractVector{<:AbstractVector{F}})::I where {F<:AbstractFloat, I<:Integer}
    js = findall(==(j), idxs)
    return js[findmin(norm, vecs[js])[2]]
end


"""
It is possible that the requested image is not present due to an insufficient neighbour
list cutoff. If this is the case then the index `0` will be returned.

"""
function locate_target_image(j::I, idxs::AbstractVector{I}, images::AbstractVector{<:AbstractVector{I}}, image::AbstractVector{I})::I where I<:Integer
    js = findall(==(j), idxs)
    idx = findfirst(i -> all(i .== image), images[js])
    return idx == nothing ? zero(I) : js[idx]
end



"""
Partition a matrix into its atom-atom blocks.
"""
function atom_block_matrix(matrix, atomic_numbers, basis_def)
    # Work out the number of orbitals present on each atom in the system
    n_orbs_atoms = get.(
        # Raise an error if a key is missing (response if key is absent)
        (() -> error("Species definition is incomplete"),),
        # Number of orbitals on each species (the dictionary to query)
        (Dict(i=>sum(2 .* j .+ 1) for (i, j) in basis_def),),
        # Atomic numbers of the species present (value to query)
        atomic_numbers)
    # Use this to partition the matrix into its atomic blocks
    return BlockArray(matrix, n_orbs_atoms, n_orbs_atoms)
end


# Three implementations of this function are required:
#   1) To deal with a single "standard" system
#   2) To deal with a periodically reduced system.
#   3) To deal with multiple systems.
# TODO:
#   - Need to add handling for on-site variant:
#       - This should ignore the lower part of the on-site block as the data
#         is a repeat of the upper part of the block.
#   - Need special handling of homo-atomic off-site systems:
#       - This should 1: combine symmetrically equivalent data. 




# The whole data representation code structure needs a significant rework 
abstract type AbstractData end

# Might be worth having the states combined into a custom AbstractConfiguration entity
# which is likely to reduce computational expense and simplify the code elsewhere.

# Add warning and discussion about the fact that this does not capture a system index
# and thus there is no way to know which system the data came from. This will eventually
# have to be added later on down the line. But it is currently not a blocker.
struct DataSet{F<:AbstractFloat, I<:Integer, S<:AbstractState} <: AbstractData
    values::Array{F, 3}
    block_indices::Matrix{I}
    cell_indices::Vector{I}
    states::Vector{Vector{S}}
end

Base.:(+)(x::T, y::Nothing) where T<:AbstractData = x
Base.:(+)(x::Nothing, y::T) where T<:AbstractData = y
function Base.:(+)(x::T, y::T) where T<:AbstractData
    return T([[getfield(x, i); getfield(y, i)] for i in fieldnames(T)]...)
end


# function Base.getindex(data_set::T, idx) where T<:AbstractData
#     return T((getfield(data_set, i)[idx, :, :] for i in fieldnames(T))...)
# end

function Base.getindex(data_set::T, idx) where T<:AbstractData
    getfieldsliced(o, n, i) = (n->n[i, repeat([:], ndims(n)-1)...])(getfield(o, n))
    return T((getfieldsliced(data_set, n, idx) for n in fieldnames(T))...)
end

Base.lastindex(data_set::AbstractData) = length(data_set)
Base.length(data_set::AbstractData) = size(data_set)[1]
Base.size(data_set::AbstractData) = size(data_set.values)

Base.adjoint(data_set::T) where T<:AbstractData = T(
        conj(permutedims(data_set.values, (1, 3, 2))),
        reverse(data_set.block_indices, dims=2),
        data_set.cell_indices,  # ← need to double check this
        [reverse.(i) for i in data_set.states])

function Base.show(io::IO, data_set::T) where T<:AbstractData
    F = eltype(data_set.values)
    I = eltype(data_set.block_indices)
    mat_shape = join(size(data_set), '×')
    print(io, "$(nameof(T)){$F, $I}($mat_shape)")
end

Parameters.ison(x::T) where T<:AbstractData = ison(x.states[1][1])

"""

Filter out data-points with sparse sub-blocks; i.e. sub-blocks in which all values are
less than the specified `threshold`.


"""
function filter_sparse(data::AbstractData, threshold::Float64)
    return data[vec(all(abs.(data.values) .>= threshold, dims=(2,3)))]
end

"""
Filter out data-points with state bond-vectors greater than the specified `distance`.
This is allows for states that will not be used to be removed during data selection
rather than evaluation.
"""
function filter_bond(data::AbstractData, distance::Float64)
    return data[[norm(i[1].rr0) <= distance for i in data.states]]
end



"""
    atomic_blk_idxs(z_i, z_j, z_s)

Indices for all atomic blocks involving between species `z_i` and `z_j`.

In effect, this yields a set of atomic index pairs representing all possible ij & ji 
pairs, where `i` & `j` are the atomic indices of atoms of species `z_i` & `z_j`
respectively. Note, only indices associated with the origin cell are returned.

# Arguments
- `z_i::Int`: first interacting species
- `z_i::Int`: second interacting species
- `z_s::Vector`: atomic numbers present in system

# Examples
```
julia> atomic_numbers = [6, 6, 1, 1, 1, 1]
julia> atomic_blk_idxs(1, 6, atomic_numbers)
16×2 Matrix{Int64}:
 3  1
 3  2
 4  1
 4  2
 5  1
 ⋮
 1  6
 2  3
 2  4
 2  5
 2  6

```

"""
function atomic_blk_idxs(z_i::I, z_j::I, z_s::Vector) where I<:Integer
    z_i_idx, z_j_idx = findall(==(z_i), z_s), findall(==(z_j), z_s)
    n, m = length(z_i_idx), length(z_j_idx)
    # Views, slices and reshape operations are used as they are faster and
    # and allocate less. 
    if z_i ≠ z_j
        res = Matrix{I}(undef, n * m * 2 , 2)
        @views let res = res[1:end ÷ 2, :]
            @views reshape(res[:, 1], (m, n)) .= z_i_idx'
            @views reshape(res[:, 2], (m, n)) .= z_j_idx
        end

        @views let res = res[1 + end ÷ 2:end, :]
            @views reshape(res[:, 1], (n, m)) .= z_j_idx'
            @views reshape(res[:, 2], (n, m)) .= z_i_idx
        end
    else
        res = Matrix{I}(undef, n * m , 2)
        @views reshape(res[:, 1], (n, m)) .= z_i_idx'
        @views reshape(res[:, 2], (n, m)) .= z_j_idx
    end
    return res
end

"""
    repeat_atomic_blk_idxs(idxs, n)

Repeat the atomic blocks indices `n` times with a new column at the start stating
which repetition the index belongs to. This is primarily intended to be used as A
way to extend an atom block index list to account for periodic images.

# Examples
julia> idxs = [10 10; 10 20; 20 10; 20 20]
julia> repeat_atomic_blk_idxs(idxs, 2)
8×3 Matrix{Int64}:
 1  10  10
 1  10  10
 1  20  20
 1  20  20
 2  10  10
 2  20  20
 2  10  10
 2  20  20
"""

function repeat_atomic_blk_idxs(idxs::Matrix{T}, n::T) where T<:Integer
    m = size(idxs, 1)
    res = Matrix{T}(undef, m * n, 3)
    @views reshape(res[:, 1], (m, n)) .= (1:n)'
    @views reshape(res[:, 2:3], (m, 2, n)) .= idxs
    return res
end

"""
    filter_on_site_idxs(idxs)

Filter out all but the on-site block indices.

# Arguments
- `idxs::Matrix{Integer}`: a matrix of atom-block indices. Each row may contain either
  a pair of atomic indices like so `[idx_i, idx_j]` or a cell index followed by a pair
  of atomic indices like so `[cell_idx, idx_i, idx_j]`. If using the latter form then
  it is assumed that a `cell_idx` value of 1 indicates the origin cell. 
   
"""
function filter_on_site_idxs(idxs::Matrix{T}) where T<:Integer
    if size(idxs, 2) == 3  # Find where atomic indices are equal & cell=1
        return idxs[idxs[:, 2] .≡ idxs[:, 3] .&& idxs[:, 1] .== 1, :]
    
    else  # Locate where atomic indices are equal
        return idxs[idxs[:, 1] .≡ idxs[:, 2], :]
    end
end

"""
    filter_off_site_idxs(idxs)

Filter out all but the off-site block indices.

# Arguments
- `idxs::Matrix{Integer}`: a matrix of atom-block indices. Each row may contain either
  a pair of atomic indices like so `[idx_i, idx_j]` or a cell index followed by a pair
  of atomic indices like so `[cell_idx, idx_i, idx_j]`. If using the latter form then
  it is assumed that a `cell_idx` value of 1 indicates the origin cell. 
   
"""
function filter_off_site_idxs(idxs::Matrix{T}) where T<:Integer
    if size(idxs, 2) == 3  # Find where atomic indices are not equal or the cell≠1.
        return idxs[idxs[:, 2] .≠ idxs[:, 3] .|| idxs[:, 1] .≠ 1, :]
    else  # Locate where atomic indices not are equal
        return idxs[idxs[:, 1] .≠ idxs[:, 2], :]
    end
end


# Return a vector specifying which atom blocks are of interest
function _blk_idx(zᵢ::T, zⱼ::T, atoms::Atoms, n_images::T=0) where T<:Integer
    # Atomic indices of all atoms of species type zᵢ and zⱼ respectively
    blk_idxs = atomic_blk_idxs(zᵢ, zⱼ, atoms.Z)

    if n_images ≠ 0  # Add image index to the blk_idx matrix for the multi-cell case.
        blk_idxs = repeat_atomic_blk_idxs(blk_idxs, n_images)
    end

    return filter_off_site_idxs(blk_idxs)  # Remove homo-atomic interaction blocks

end

function _blk_idx(zᵢ::T, atoms::Atoms, n_images::T=0) where T<:Integer
    blk_idxs = filter_on_site_idxs(atomic_blk_idxs(zᵢ, zᵢ, atoms.Z))

    if n_images ≠ 0  # Add image index to the blk_idx matrix for the multi-cell case.
        blk_idxs = repeat_atomic_blk_idxs(blk_idxs, n_images)
    end

    return filter_on_site_idxs(blk_idxs)  # Remove hetro-atomic interactions blocks

end


# The `_assign_sub_blocks!` methods are responsable for collecting sub-blocks from the
# array `from` and placing them into the array `to`. Once the data has been changed so
# that it is column major then this will no longer be required.
function _assign_sub_blocks!(from::Matrix{T}, to::AbstractArray{T, 3}, blk_idxs, blk_starts, sliceᵢ, sliceⱼ) where T
    for (n, (row, col)) in enumerate(eachrow(blk_idxs))
        to[n, :, :] = from[blk_starts[row] .+ sliceᵢ, blk_starts[col] .+ sliceⱼ]
    end
end

function _assign_sub_blocks!(from::AbstractArray{T, 3}, to::AbstractArray{T, 3}, blk_idxs, blk_starts, sliceᵢ, sliceⱼ) where T
    for (n, (cell, row, col)) in enumerate(eachrow(blk_idxs))
        to[n, :, :] = from[cell, blk_starts[row] .+ sliceᵢ, blk_starts[col] .+ sliceⱼ]
    end
end


# Returns a vector specifying the index a which each atom-block starts along with a pair
# of slicers that can extract the requested sub-block from its associated atom-block.
function _starts_and_slices(zᵢ, zⱼ, sᵢ, sⱼ, atoms, basis_def)

    # Number of oritals present on each shell of each species
    n_orbs_s = Dict(i=> 2 .* j .+ 1 for (i, j) in basis_def)
    # Total number of orbitals present on each species
    n_orbs_a = Dict(i=>sum(j) for (i, j) in n_orbs_s)

    # Indicies indicating where each atomic block starts
    starts = (i -> cumsum(i) .- i)(getindex.(Ref(n_orbs_a), getfield.(atoms.Z, :z)))

    # Number of orbitals present on the specified shells
    nᵢ, nⱼ = n_orbs_s[zᵢ][sᵢ], n_orbs_s[zⱼ][sⱼ]

    # These StepRange pairs are used to gather sub-blocks from a target atom-block.
    # For example `sub_block = atom_block[rₛ, cₛ]`.
    sliceᵢ, sliceⱼ = let aᵢ = cumsum(n_orbs_s[zᵢ]), aⱼ = cumsum(n_orbs_s[zⱼ])
        aᵢ[sᵢ] - nᵢ + 1:1:aᵢ[sᵢ], aⱼ[sⱼ] - nⱼ + 1:1:aⱼ[sⱼ]
    end

    return starts, sliceᵢ, sliceⱼ
end

function collect_matrix_data(matrix::AbstractArray{T, N}, zᵢ::I, zⱼ::I, sᵢ::I, sⱼ::I, atoms, basis_def) where {T<:AbstractFloat, N, I<:Integer}
    n = N ≡ 2 ? 0 : size(matrix, 1)::Int64
    blk_idxs = _blk_idx(zᵢ, zⱼ, atoms, n)
    # Remove symmetrically equivilent datapoints when appropriate
    if zᵢ ≡ zⱼ && sᵢ == sⱼ
        blk_idxs = blk_idxs[blk_idxs[:, end-1] .<= blk_idxs[:, end], :]
    end

    blk_starts, sliceᵢ, sliceⱼ = _starts_and_slices(zᵢ, zⱼ, sᵢ, sⱼ, atoms, basis_def)

    nᵢ, nⱼ = basis_def[zᵢ][sᵢ] * 2 + 1, basis_def[zⱼ][sⱼ] * 2 + 1

    data = Array{T, 3}(undef, size(blk_idxs, 1), nᵢ , nⱼ)

    _assign_sub_blocks!(matrix, data, blk_idxs, blk_starts, sliceᵢ, sliceⱼ)
    return data, blk_idxs
end


function collect_matrix_data(matrix::AbstractArray{T, N}, zᵢ::I, sᵢ::I, sⱼ::I, atoms, basis_def) where {T<:AbstractFloat, N, I<:Integer}
    n = N ≡ 2 ? 0 : size(matrix, 1)::Int64
    blk_idxs = _blk_idx(zᵢ, atoms, n)

    blk_starts, sliceᵢ, sliceⱼ = _starts_and_slices(zᵢ, zᵢ, sᵢ, sⱼ, atoms, basis_def)

    nᵢ, nⱼ = basis_def[zᵢ][sᵢ] * 2 + 1, basis_def[zᵢ][sⱼ] * 2 + 1

    data = Array{T, 3}(undef, size(blk_idxs, 1), nᵢ , nⱼ)

    _assign_sub_blocks!(matrix, data, blk_idxs, blk_starts, sliceᵢ, sliceⱼ)
    return data, blk_idxs
end

function collect_data(matrix::AbstractMatrix{T}, basis::Basis, atoms::Atoms,
    basis_def::Dict; tol::Union{Nothing, T}=nothing) where T
    blocks, blk_idxs = collect_matrix_data(matrix, basis.id..., atoms, basis_def)
    if !isnothing(tol)
        mask = vec(all(abs.(blocks) .>= tol, dims=(2,3)))
        blocks = blocks[mask, :, :]
        blk_idxs = blk_idxs[mask, :]
    end
    block_i = blk_idxs
    cell_i = zeros(valtype(block_i), size(block_i, 1))
    if ison(basis)
        states = get_states.(block_i[:, 1], (atoms,))
    else
        states = get_states.(eachcol(block_i)..., (atoms,), (envelope(basis),))
    end
    return DataSet(blocks, block_i, cell_i, states)
end


function collect_data(matrix::AbstractArray{T, 3}, basis::Basis, atoms::Atoms,
    basis_def::Dict, images::AbstractMatrix{I}; tol::Union{Nothing, T}=nothing) where {T, I<:Integer}
    blocks, blk_idxs = collect_matrix_data(matrix, basis.id..., atoms, basis_def)
    if !isnothing(tol)
        mask = vec(all(abs.(blocks) .>= tol, dims=(2,3)))
        blocks = blocks[mask, :, :]
        blk_idxs = blk_idxs[mask, :]
    end
    block_i, cell_i = blk_idxs[:, 2:end], blk_idxs[:, 1]

    if ison(basis)
        states = get_states.(block_i[:, 1], (atoms,))
    else
        @views images = eachrow(images[cell_i, :])
        states = get_states.(eachcol(block_i)..., (atoms,), (envelope(basis),), images)
    end

    return DataSet(blocks, block_i, cell_i, states)
end


function collect_data_from_hdf5(src::Group, basis::Basis, target::Symbol;
    tol::Union{Nothing, T}=nothing) where T<:AbstractFloat
    # Currently this inefficient as it requires each dataset to be reloaded for
    # each and every basis.
    # This is only useful for gathering a small subset of data for testing.
    @assert target in [:H, :S]
    return collect_data(
        target ≡ :H ? load_hamiltonian(src) : load_overlap(src),
        basis,
        load_atoms(src),
        load_basis_set_definition(src),
        gamma_only(src) ? nothing : load_cell_translations(src);
        tol=tol)
end


function locate_blocks(z₁::I, atoms::Atoms) where I<:Integer
    atomic_numbers = getfield.(atoms.Z, :z)
    zᵢ = findall(atomic_numbers .== z₁)
    return reduce(vcat, [i i] for i in zᵢ)
end

function locate_blocks(z₁::I, z₂::I, atoms::Atoms) where I<:Integer
    atomic_numbers = getfield.(atoms.Z, :z)  
    zᵢ = findall(atomic_numbers .== z₁)
    zⱼ = findall(atomic_numbers .== z₂)
    return reduce(vcat, [i j] for i in zᵢ for j in zⱼ)
end

function upper_blocks(idxs::Matrix{I}) where I<:Integer
    return idxs[findall(idxs[:, 1] .>= idxs[:, 2]), :]
end

function off_site_blocks(idxs::Matrix{I}) where I<:Integer
    return idxs[findall(idxs[:, 1] .!= idxs[:, 2]), :]
end

function on_site_blocks(idxs::Matrix{I}) where I<:Integer
    return idxs[findall(idxs[:, 1] .== idxs[:, 2]), :]
end

# Note that this is mostly a temporary function.
function gather_matrix_blocks(matrix::Matrix{T}, blocks::Matrix, s₁, s₂, atoms::Atoms, basis_def) where T<:AbstractFloat

    atomic_numbers = getfield.(atoms.Z, :z)

    n_orbs_s = Dict(i=> 2 .* j .+ 1 for (i, j) in basis_def)
    n_orbs_a = Dict(i=>sum(j) for (i, j) in n_orbs_s)

    z₁, z₂ = atomic_numbers[blocks[1, :]]
    # z₁, z₂ = atomic_numbers[blocks[1, 1]], atomic_numbers[blocks[1, 1]]
    n₁, n₂ = n_orbs_s[z₁][s₁], n_orbs_s[z₂][s₂]

    
    startᵢ = (i -> cumsum(i) .- i)(getindex.(Ref(n_orbs_a), atomic_numbers))
    
    f = (zₖ, sₖ, nₖ) -> (a -> a[sₖ] - nₖ + 1:1:a[sₖ])(cumsum(n_orbs_s[zₖ]))
    rₛ, cₛ = f.((z₁, z₂), (s₁, s₂), (n₁, n₂))

    # While it is slower to use undefined vectors however doing otherwise
    # (i.e. data = Matrix{T}[]) breaks the data link.
    data = []

    for (rᵢ, cᵢ) in  eachrow(blocks)
        @views sub_block = matrix[startᵢ[rᵢ] .+ rₛ, startᵢ[cᵢ] .+ cₛ]
        append!(data, Ref(sub_block))
    end

    return data
end


# """
#     locate_atomic_blocks(z₁, z₂, atoms)

# Matrix of indices corresponding to atomic blocks between species z₁ and z₂.

# Note, symmetrically equivalent blocks are **not** gathered; i.e. z₁, z₂ = 1, 6 does not
# return the same atomic block indices as z₁, z₂ = 6, 1.
# """
# function locate_atomic_blocks(z₁::I, z₂::I, atoms::Atoms) where I<:Integer
#     atomic_numbers = getfield.(atoms.Z, :z)
#     i, j = findall(atomic_numbers .== z₁), findall(atomic_numbers .== z₂)
#     return [repeat(i, inner=length(j));; repeat(j, outer=length(i))]
# end

# """
# Extract and return atomic-block indices which correspond to off-site interactions.
# """
# function extract_off_site_blocks(idxs::Matrix{I}) where I<:Integer
#     return indices[findall(idxs[:, 1] .!= idxs[:, 2]), :]
# end


# """
# Extract and return atomic-block indices which correspond to on-site interactions.
# """
# function extract_off_site_blocks(idxs::Matrix{I}) where I<:Integer
#     return indices[findall(idxs[:, 1] .== idxs[:, 2]), :]
# end

# function extract_symmetrically_unique(idxs::Matrix{I}) where I<:Integer
    
#     #reduce(vcat, a)
#     #return 
# end



# """

# Gather the requested sub-block from specific blocks of a source matrix.

# """
# function gather_sub_blocks(source::Matrix{F}, blocks::Matrix{I}, sub_block::Tuple{I, I}, basis_def, atoms) where {F, I<:Integer}
    
    
#     # values = Array{valtype(source), 3}(size(blocks, 1), n, m)
# end

# function assign_sub_blocks(target, blocks, sub_block, basis_def, atoms, values)
# end

# locate_sub_blocks()
# gather_sub_blocks()
# assign_sub_blocks()

end

# """
# # Warnings
# This code is not production ready

# Note that data will be collected from all supplied matrix images. However, only data from the
# origin cell is actually meaningful for on-site interactions.  
# While data will be gathered from all k-point images  

# # Developers Notes
# Although this is operational it is extremely convoluted and thus requires refactoring.
# Multiple dispatch should be used to distinguish between the supercell & k-grid variants.
# Repeated code blocks should be abstracted either to user accessible functions or code
# snippet macro blocks. Would probably be better to split this into on and off site variants.
# """
# function collect_matrix_data(matrix::Array{T, N₁}, interaction::NTuple{N₂, I}, atoms::Atoms, basis_def)::Tuple{Array{T, 3}, Matrix{I}, Vector{I}} where {T<:AbstractFloat, N₁, N₂, I<:Integer}
#     # For k-point calculations with homo-atomic-blocks only the origin cell is relevant;
#     # thus all other cells are removed. Note that this assumes that the first cell is the origin cell. 
#     if ndims(matrix) ≡ 3 && length(interaction) == 3
#         @views matrix = matrix[1, :, :]
#     end

#     atomic_numbers = getfield.(atoms.Z, :z)
    
#     # Step 1.1: Setup
#     # Calculate the number of orbitals present on each shell and on each species.
#     n_orbs_s = Dict(i=> 2 .* j .+ 1 for (i, j) in basis_def)
#     n_orbs_a = Dict(i=>sum(j) for (i, j) in n_orbs_s)

#     # Extract various static properties.
#     z₁, z₂ = interaction[1], interaction[end-2] # ← atomic numbers
#     s₁, s₂ = interaction[end-1:end] # ← shell numbers
#     n₁, n₂ = n_orbs_s[z₁][s₁], n_orbs_s[z₂][s₂] # ← number of orbitals on each shell

#     # Step 1.2: Atom-block index pair list generation
#     # Get the atomic indices of specified atoms.   
#     zᵢ, zⱼ = (z -> findall(==(z), atomic_numbers)).((z₁, z₂))
    
#     # Identify which atom-blocks should be targeted.
#     if length(interaction) == 3 # On-site: diagonal blocks
#         blocks = collect((i, j) for (i, j) in zip(zᵢ, zⱼ))
#     elseif z₁ ≠ z₂ # Off-site (hetro-atomic): off-diagonal blocks
#         blocks = collect((i, j) for i in zᵢ for j in zⱼ)
#     else # Off-site (homo-atomic): off-diagonal blocks only from the upper triangle; but,
#          # included symmetrically equivalent blocks from the lower triangle if s₁≠s₂.
#         blocks = collect((i, j) for i in zᵢ for j in zⱼ if i<j || (i>j && s₁ ≠ s₂))
#     end

#     # Step 1.3: Collect the desired sub-blocks from the specified atom-blocks 
#     # Identify where each atomic-block starts
#     startᵢ = (i -> cumsum(i) .- i)(getindex.(Ref(n_orbs_a), atomic_numbers))
    
#     # A StepRange pair is used to gather sub-blocks from their atomic-blocks;
#     # i.e. `sub_block = block[rₛ, cₛ]`. 
#     f = (zₖ, sₖ, nₖ) -> (a -> a[sₖ] - nₖ + 1:1:a[sₖ])(cumsum(n_orbs_s[zₖ]))
#     rₛ, cₛ = f.((z₁, z₂), (s₁, s₂), (n₁, n₂))  # ← row/column StepRange pair

#     # If the matrix is 3D (k-point calculations); there will be cellsₙ * n_sub_blocks
#     # data-points rather than just n_sub_blocks.
#     cellsₙ = ndims(matrix) ≡ 2 ? 1 : size(matrix)[1]
    
#     # Extra dimension required for slicing 3D matrices
#     dim = ndims(matrix) ≡ 3 ? (:,) : () 

#     # Construct matrix to hold the results
#     data = Array{eltype(matrix), 3}(undef, length(blocks) * cellsₙ, n₁, n₂)
    
#     # Gather the target sub-blocks and append them to the data array. Note that the
#     # assignment operation must work for both the 2D and 3D "matrix" cases.
#     for (n, (rᵢ, cᵢ)) in enumerate(blocks)
#         @views sub_block = matrix[dim..., startᵢ[rᵢ] .+ rₛ, startᵢ[cᵢ] .+ cₛ]
#         data[(-cellsₙ + 1 + n * cellsₙ):(n * cellsₙ), :, :] = sub_block
#     end

#     # Collect and return the cell numbers (normally 1 unless matrix is 3D)
#     cells = [c for _ in blocks for c in 1:cellsₙ]

#     # Add the cell number to the block for the multi-cell instance
#     blocks = [b for b in blocks for _ in 1:cellsₙ]

#     return data, collect(reduce(hcat, collect.(blocks))'), cells

# end

# function collect_data(matrix::AbstractArray, basis::Basis, atoms::Atoms,
#     basis_def::Dict, images::Union{Matrix, Nothing}=nothing)    
#     blocks, block_i, cell_i = collect_matrix_data(matrix, basis.id, atoms, basis_def)
#     if ison(basis)
#         states = get_states.(block_i[:, 1], (atoms,))
#     else
#         if isnothing(images)
#             states = get_states.(eachcol(block_i)..., (atoms,), (envelope(basis),))
#         else
#             @views images = eachrow(images[cell_i, :])
#             states = get_states.(eachcol(block_i)..., (atoms,), (envelope(basis),), images)
#         end
#     end

#     return DataSet(blocks, block_i, cell_i, states)
# end