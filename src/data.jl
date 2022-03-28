module DataManipulation
using BlockArrays
using HDF5
using HDF5: Group
using NeighbourLists
using ACE: BondEnvelope, State, PositionState
using Base
using JuLIP
using JuLIP: Atoms
using JuLIP.Utils: project_min
using ACEhamiltonians

export collect_data, collect_data_from_hdf5, DataSet
# using ACEhamiltonians.Common: BasisDef



# Todo:
#   - This code should be partitioned into separate, more relevant, modules. Such as an
#     io module and a data manipulation module. If possible much of this should be
#     generalised and abstracted to a separate ACE package.
#   - The io module should have HDF5 and JSON sub-modules.
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
    - Rename csv to image.
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
    pair_list = JuLIP.neighbourlist(atoms, r)

    # Extract environment about each relevant atom from the pair list. These will be tuples
    # of the form: (atomic-index, relative-position)
    environments = NeighbourLists.neigs(pair_list, i)

    # zs = getindex.(Ref(atoms.Z), map(i->i[1], environments))  # ← to be added later
    return PositionState.(environments[2])
end


"""
    get_states(i, j, atoms[; ctv, r])

Construct a sate vector describing the local environment about a target bond.

# Arguments
- `i::Integer`: atomic index of the first bond atom.
- `j::Integer`: atomic index of the second bond atom.
- `atoms::Atoms`: the atomic system to which the bond relates.
- `envelope::BondEnvelope`: bond envelope used to refine local environment.
- `ctv::Vector{Integer}`: vector specifying in which periodic cell atom `j` resides.
   This defaults to the origin cell, i.e. [0, 0, 0].
- `r::AbstractFloat`: maximum view distance for the environment, defaults to 20 Å.

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

# Warnings
The environment distance cutoff `r` acts only as a minimum distance. As such is it
possible that species beyond this distances may be returned. This is because the
neighbour list is cached by the `Atoms` object to improve performance.
"""
function get_states(i::I, j::I, atoms::Atoms, envelope::BondEnvelope,
    ctv::AbstractArray{I}=Vector{I}([0, 0, 0]); r::F=20.0) where {I<:Integer, F<:AbstractFloat}
    
    # Single system no translation vectors
    pair_list = JuLIP.neighbourlist(atoms, r) # ← TODO: REMOVE
    
    # Calculate the bond vector between i and the closest periodic image of j. Then shift
    # it so that it points the requested periodic image, as per the csv.
    rr0 = project_min(atoms, atoms.X[j] - atoms.X[i])
    rr0 += (adjoint(ctv .* atoms.pbc) * atoms.cell).parent
    
    # Bond state; "rr"=position of i relative j, "rr0"=bond vector & "be" indicates
    # that this is a bond (rr==rr0 here).
    bond_state = State(rr=rr0, rr0=rr0, be=:bond)

    # Get the indices & distance vectors of atoms neighbouring i
    idxs, vecs = NeighbourLists.neigs(pair_list, i)

    # Locate j; done to avoid accidentally including it as an environmental atom
    j_idx = findfirst(!=(rr0), vecs)

    # Environmental atoms states, "rr0"=bond=vector, "rr"=atom's position relative to i &
    # "be" indicates that this is an environmental state.
    env_states = (x -> State(rr=x, rr0=rr0, be=:env)).(vecs[1:end .!= j_idx])
    # Cull states outside of the bond envelope. A double filter is required as inbuilt
    # filter operation deviates from standard julia behaviour.
    env_states = Base.filter(x -> filter(envelope, x), env_states)

    return [bond_state; env_states]

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



# macro snippet(name, definition)
#     return quote
#         macro $(esc(name))()
#             esc($(Expr(:quote, definition)))
#         end
#     end
# end

# function gather_blocks(matrix, shells)

#     return

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

"""
# Warnings
This code is not production ready

Note that data will be collected from all supplied matrix images. However, only data from the
origin cell is actually meaningful for on-site interactions.  
While data will be gathered from all k-point images  

# Developers Notes
Although this is operational it is extremely convoluted and thus requires refactoring.
Multiple dispatch should be used to distinguish between the supercell & k-grid variants.
Repeated code blocks should be abstracted either to user accessible functions or code
snippet macro blocks.
"""
function collect_matrix_data(matrix::Array, interaction, atoms::Atoms, basis_def)
    # For k-point calculations only the origin cell is relevant; thus all other cells are
    # removed. Note that this assumes that the first cell is the origin cell. 
    if ndims(matrix) ≡ 3 && length(interaction) == 3
        @views matrix = matrix[1, :, :]
    end

    atomic_numbers = getfield.(atoms.Z, :z)
    
    # Step 1.1: Setup
    # Calculate the number of orbitals present on each shell and on each species.
    n_orbs_s = Dict(i=> 2 .* j .+ 1 for (i, j) in basis_def)
    n_orbs_a = Dict(i=>sum(j) for (i, j) in n_orbs_s)

    # Extract various static properties.
    z₁, z₂ = interaction[1], interaction[end-2] # ← atomic numbers
    s₁, s₂ = interaction[end-1:end] # ← shell numbers
    n₁, n₂ = n_orbs_s[z₁][s₁], n_orbs_s[z₂][s₂] # ← number of orbitals on each shell

    # Step 1.2: Atom-block index pair list generation
    # Get the atomic indices of specified atoms.   
    zᵢ, zⱼ = (z -> findall(==(z), atomic_numbers)).((z₁, z₂))
    
    # Identify which atom-blocks should be targeted.
    if length(interaction) == 3 # On-site: diagonal blocks
        blocks = collect((i, j) for (i, j) in zip(zᵢ, zⱼ))
    elseif z₁ ≠ z₂ # Off-site (hetro-atomic): off-diagonal blocks
        blocks = collect((i, j) for i in zᵢ for j in zⱼ)
    else # Off-site (homo-atomic): off-diagonal blocks only from the upper triangle; but,
         # included symmetrically equivalent blocks from the lower triangle if s₁≠s₂.
        blocks = collect((i, j) for i in zᵢ for j in zⱼ if i>j || (i<j && s₁ ≠ s₂))
    end

    # Step 1.3: Collect the desired sub-blocks from the specified atom-blocks 
    # Identify where each atomic-block starts
    startᵢ = (i -> cumsum(i) .- i)(getindex.(Ref(n_orbs_a), atomic_numbers))
    
    # A StepRange pair is used to gather sub-blocks from their atomic-blocks;
    # i.e. `sub_block = block[rₛ, cₛ]`. 
    f = (zᵢ, sᵢ, nᵢ) -> (a -> a[sᵢ] - nᵢ + 1:1:a[sᵢ])(cumsum(n_orbs_s[zᵢ]))
    rₛ, cₛ = f.((z₁, z₂), (s₁, s₂), (n₁, n₂))  # ← row/column StepRange pair

    # If the matrix is 3D (k-point calculations); there will be cellsₙ * n_sub_blocks
    # data-points rather than just n_sub_blocks.
    cellsₙ = ndims(matrix) ≡ 2 ? 1 : size(matrix)[1]
    
    # Extra dimension required for slicing 3D matrices
    dim = ndims(matrix) ≡ 3 ? (:,) : () 

    # Construct matrix to hold the results
    data = Array{eltype(matrix), 3}(undef, length(blocks) * cellsₙ, n₁, n₂)
    
    # Gather the target sub-blocks and append them to the data array. Note that the
    # assignment operation must work for both the 2D and 3D "matrix" cases.
    for (n, (rᵢ, cᵢ)) in enumerate(blocks)
        @views sub_block = matrix[dim..., startᵢ[rᵢ] .+ rₛ, startᵢ[cᵢ] .+ cₛ]
        data[(-cellsₙ + 1 + n * cellsₙ):(n * cellsₙ), :, :] = sub_block
    end

    # Collect and return the cell numbers (normally 1 unless matrix is 3D)
    cells = [c for _ in blocks for c in 1:cellsₙ]

    # Add the cell number to the block for the multi-cell instance
    blocks = [b for b in blocks for _ in 1:cellsₙ]


    return data, collect(reduce(hcat, collect.(blocks))'), cells

end

# Add warning and discussion about the fact that this does not capture a system index
# and thus there is no way to know which system the data came from. This will eventually
# have to be added later on down the line. But it is currently not a blocker.
struct DataSet{F<:AbstractFloat, I<:Integer, S<:State}
    values::Array{F, 3}
    block_indices::Matrix{I}
    cell_indices::Vector{I}
    states::Vector{Vector{S}}
end

Base.:(+)(x::DataSet{F, I, S}, y::Nothing) where {F, I, S} = x
Base.:(+)(x::Nothing, y::DataSet{F, I, S}) where {F, I, S} = y
Base.:(+)(x::DataSet{F, I, S}, y::DataSet{F, I, S}) where {F, I, S} = DataSet(
    [x.values; y.values], [x.block_indices; y.block_indices],
    [x.cell_indices; y.cell_indices], [x.states; y.states])

Base.getindex(data_set::DataSet, idx::O) where O<:OrdinalRange = DataSet(
    data_set.values[idx, :, :],
    data_set.block_indices[idx, :],
    data_set.cell_indices[idx],
    data_set.states[idx])

Base.lastindex(data_set::DataSet) = length(data_set)
Base.length(data_set::DataSet) = size(data_set)[1]
Base.size(data_set::DataSet) = size(data_set.values)




function Base.show(io::IO, data_set::DataSet)
    F = eltype(data_set.values)
    I = eltype(data_set.block_indices)
    mat_shape = join(size(data_set), '×')
    print(io, "DataSet{$F, $I}($mat_shape)")
end


function collect_data(matrix::AbstractArray, basis::Basis, atoms::Atoms,
    basis_def::Dict, images::Matrix=[1 1 1])
    blocks, block_i, cell_i = collect_matrix_data(matrix, basis.id, atoms, basis_def)

    if ison(basis)
        states = get_states.(block_i[:, 1], (atoms,))
    else
        @views images = eachrow(images[cell_i, :])
        states = get_states.(eachcol(block_i)..., (atoms,), (envelope(basis),), images)
    end

    return DataSet(blocks, block_i, cell_i, states)
end


function collect_data_from_hdf5(src::Group, basis::Basis, target::Symbol)
    # Currently this inefficient as it requires each dataset to be reloaded for
    # each and every basis.
    # This is only useful for gathering a small subset of data for testing.
    @assert target in [:H, :S]
    return collect_data(
        target ≡ :H ? load_hamiltonian(src) : load_overlap(src),
        basis,
        load_atoms(src),
        load_basis_set_definition(src),
        gamma_only(src) ? [1 1 1 ] : load_cell_translations(src))
end


# locate_sub_blocks()
# gather_sub_blocks()
# assign_sub_blocks()

end