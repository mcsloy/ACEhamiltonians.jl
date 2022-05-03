module Fitting2
using HDF5, ACE, ACEbase, ACEhamiltonians, StaticArrays, Statistics, LinearAlgebra
using HDF5: Group
using JuLIP: Atoms
using ACE: ACEConfig, evaluate, scaling, AbstractState
using ACEhamiltonians.Fitting: evaluateval_real, solve_ls
using ACEhamiltonians.Common: number_of_orbitals
using ACEhamiltonians.DataManipulation: get_states, locate_blocks, upper_blocks, off_site_blocks, gather_matrix_blocks
using ACEhamiltonians.Bases: envelope
using ACEhamiltonians.DataIO: load_hamiltonian_gamma, load_hamiltonian_gamma

export assemble_ls_new, fit!, predict, predict!


# Todo:
#   - Need to make sure that the acquire_B! function used by ACE does not actually modify the
#     basis function. Otherwise there may be some issues with sharing basis functions.
#   - ACE should be modified so that `valtype` inherits from Base. This way there should be
#     no errors caused when importing it.
#   - Remove hard coded matrix type from the predict function. 

function evaluateval_real_new(Aval)

    n₁, n₂ = size(Aval[1])
    ℓ₁, ℓ₂ = Int((n₁ - 1) / 2), Int((n₂ - 1) / 2)

    # allocate Aval_real
    Aval_real = [zeros(ComplexF64, n₁, n₂) for i = 1:length(Aval)]
    # reconstruct real A
    # TODO: I believe that there must exist a matrix form... But I will keep it for now...
    for k=1:length(Aval)

        A = Aval[k].val
        for i=1:n₁, j=1:n₂
            # Magnetic quantum numbers
            m₁, m₂ = i - ℓ₁ - 1, j - ℓ₂ - 1
            
            val = A[i,j]
            
            if m₁ < 0
                val -= (-1)^(m₁) * A[end-i+1,j]
            elseif m₁ > 0
                val += (-1)^(m₁) * A[end-i+1,j]
            end

            if (m₁ > 0 < m₂) || (m₁ < 0 > m₂)
                val += (-1)^(m₁ + m₂) * A[end-i+1, end-j+1]
            elseif (m₁ > 0 > m₂) || (m₁ < 0 < m₂)
                val -= (-1)^(m₁ + m₂) * A[end-i+1, end-j+1]
            end
            if m₂ < 0
                val -= (-1)^(m₂) * A[i, end-j+1]
            elseif m₂ > 0
                val += (-1)^(m₂) * A[i, end-j+1]
            end

            # This could be applied as a matrix to the full block
            s = ((m₁ >= 0) && (m₂ < 0)) ? -1 : 1 # Account for sign flip until mess is sorted
            plane = (m₁ >= 0) ⊻ (m₂ >= 0) ? im : 1
            scale = m₁ == 0 ? (m₂ == 0 ? 1 : 1/√2) : (m₂ == 0 ? 1/√2 : 1/2)

            Aval_real[k][i,j] = scale * plane * (s * (val))
        end

        # Prefactor "scale" could be applied here as a matrix operation
    end
    #return Aval_real
    if norm(Aval_real - real(Aval_real))<1e-12
        return real(Aval_real)
    else
        error("norm = $(norm(Aval_real - real(Aval_real))), please recheck...")
    end
end

# +1 * A[k,j] always

# -1 * s₁*A[α,j] m₁<0
#  0 * s₁*A[α,j] m₁=0
# +1 * s₁*A[α,j] m₁>0

# -1 * s₂*A[i,β] m₂<0
#  0 * s₂*A[i,β] m₂=0
# +1 * s₂*A[i,β] m₂>0

# -1 * s₃*A[α,β] m₁ > 0 > m₂ or m₁ < 0 < m₂
#  0 * s₃*A[α,β] m₁=0 or m₂=0
# +1 * s₃*A[α,β] m₁ > 0 < m₂) or (m₁ < 0 > m₂

# m₁<0:
#   - m₂<0 : 1/2  (A)
#   - m₂=0 : i/√2 (B)
#   - m₂>0 : i/2  (C)

# m₁=0:
#   - m₂<0 : i/√2 (D)
#   - m₂=0 : 1    (E)
#   - m₂>0 : 1/√2 (F)

# m₁>0:
#   - m₂<0 : i/2  (G)
#   - m₂=0 : 1/√2 (H)
#   - m₂>0 : 1/2  (I)

# if kc₁ && jc₁ # A (1/2)
#     val += - s₁*A[α,j] - s₂*A[k,β] + s₃*A[α,β]
# elseif kc₁ && jc₂ # B (im/sqrt(2))
#     val += - s₁*A[α,j]
# elseif kc₁ && jc₃ # C (im/2)
#     val += - s₁*A[α,j] + s₂*A[k,β] - s₃*A[α,β]

# elseif kc₂ && jc₁ # D (im/sqrt(2))
#     val += - s₂*A[k,β]
# # elseif kc₂ && jc₂ # E (1)
# #     val = A[k,j]
# elseif kc₂ && jc₃ # F (1/sqrt(2))
#     val += + s₂*A[k,β]

# elseif kc₃ && jc₁ # G (im/2)
#     val += + s₁*A[α,j] - s₂*A[k,β] - s₃*A[α,β]
# elseif kc₃ && jc₂ # H (1/sqrt(2))
#     val += + s₁*A[α,j]
# elseif kc₃ && jc₃ # I (1/2)
#     val += + s₁*A[α,j] + s₂*A[k,β] + s₃*A[α,β]
    
# end



function assemble_ls_new(basis::Basis, data::DataSet, zero_mean::Bool=false)
    # This will be rewritten once the other code has been refactored.

    # Should `A` not be constructed using `acquire_B!`?

    n₀, n₁, n₂ = size(data)
    # Currently the code desires "A" to be an X×Y matrix of Nᵢ×Nⱼ matrices, where X is
    # the number of sub-block samples, Y is equal to `size(bos.basis.A2Bmap)[1]`, and
    # Nᵢ×Nⱼ is the sub-block shape; i.e. 3×3 for pp interactions. This may be refactored
    # at a later data if this layout is not found to be strictly necessary.
    cfg = ACEConfig.(data.states)
    Aval = evaluate.(Ref(basis.basis), cfg)
    A = permutedims(reduce(hcat, evaluateval_real_new.(Aval)), (2, 1))
    # A = permutedims(reduce(hcat, evaluateval_real.(Aval)), (2, 1))
    
    Y = [data.values[i, :, :] for i in 1:n₀]

    # Calculate the mean value x̄
    if !zero_mean && n₁ ≡ n₂ && ison(basis) 
        x̄ = mean(diag(mean(Y)))*I(n₁)
    else
        x̄ = zeros(n₁, n₂)
    end

    Y .-= Ref(x̄)
    return A, Y, x̄
end


function fit!(basis::Basis, data::DataSet, zero_mean::Bool=false)
    # Lambda term should not be hardcoded to 1e-7!

    # Get the basis function's scaling factor (?)
    Γ = Diagonal(scaling(basis.basis, 2))

    # Setup the least squares problem
    Φ, Y, x̄ = assemble_ls_new(basis, data, zero_mean)
    
    # Assign the mean value to the basis set
    basis.mean = x̄

    # Solve the least squares problem and get the coefficients
    basis.coefficients = collect(solve_ls(Φ, Y, 1e-7, Γ, "LSQR"))

    nothing
end


"""
Todo:
    - It will be important to check that the relevant data exists before trying to
      extract it; i.e. don't bother trying to gather carbon on-site data from a H2
      system etc.
    - Currently the basis set definition is loaded for the first system under the
      assumption that it constant across all systems. However, this will break down
      if different species are present in each system, i.e. if the first system is
      H2 but the second is CH4 then the carbon basis set will be absent.
    - The approach taken here limits io overhead by reducing redundant load operations.
      However, this will likely use considerably more memory.
"""
function fit!(model::Model, systems::Vector{Group}, target::Symbol)
    # Section 1: Gather the data

    @assert target in [:H, :S]
    on_site_data = Dict{Basis, DataSet}()
    off_site_data = Dict{Basis, DataSet}()

    add_data!(dict, key, data) = dict[key] = data + getkey(dict, key, nothing) 

    # get_matrix = target ≡ :H ? load_hamiltonian : load_overlap # Todo: Uncomment
    get_matrix = load_hamiltonian_gamma
    
    # Loop over the specified systems
    for system in systems
        # Load the required data from the database entry
        matrix = get_matrix(system)
        atoms = load_atoms(system)
        # images = gamma_only(system) ? [0 0 0] : load_cell_translations(system)
        

        # Loop over the on site bases and collect the appropriate data
        for basis in values(model.on_site_bases)
            # data_set = collect_data(matrix, basis, atoms, model.basis_definition, images)
            data_set = collect_data(matrix, basis, atoms, model.basis_definition)
            add_data!(on_site_data, basis, data_set)
        end 

        # Repeat for the off-site models
        for basis in values(model.off_site_bases)
            # data_set = collect_data(matrix, basis, atoms, model.basis_definition, images)
            data_set = collect_data(matrix, basis, atoms, model.basis_definition)
            add_data!(off_site_data, basis, data_set)
        end         
    end

    # Fit the on-site models
    for (basis, data_set) in on_site_data
        fit!(basis, data_set)
    end

    for (basis, data_set) in off_site_data
        fit!(basis, data_set)
    end
end


# function get_data(model::Model, systems::Vector{Group}, target::Symbol)
#     # Section 1: Gather the data

#     @assert target in [:H, :S, :Hg]
#     on_site_data = Dict{Basis, DataSet}()
#     off_site_data = Dict{Basis, DataSet}()

#     add_data!(dict, key, data) = dict[key] = data + getkey(dict, key, nothing) 

#     get_matrix = target ≡ :H ? load_hamiltonian : load_overlap

#     # Loop over the specified systems
#     for system in systems
#         # Load the required data from the database entry
#         matrix = get_matrix(system)
#         atoms = load_atoms(system)
#         images = gamma_only(system) ? [1 1 1] : load_cell_translations(system)

#         # Loop over the on site bases and collect the appropriate data
#         for basis in values(model.on_site_bases)
#             data_set = collect_data(matrix, basis, atoms, model.basis_definition, images)
#             add_data!(on_site_data, basis, data_set)
#         end 

#         # Repeat for the off-site models
#         for basis in values(model.off_site_bases)
#             data_set = collect_data(matrix, basis, atoms, model.basis_definition, images)
#             add_data!(off_site_data, basis, data_set)
#         end         
#     end

#     return on_site_data, off_site_data
# end

# Todo:
#   - Discuss why there are so many predict functions here:
#       - One for placing results in to standard array
#       - One for creating the array first then passing off to the previous method 


# The basis specific predict functions should come in two forms andversions 

# Predict functions for: single, batch by collection, batch by aggregation.
# Forms for pre-supplied arrays and for internal construction.


# Fill a single supplied array with the results from multiple states
function predict!(values::Array{F, 3}, basis::Basis, states::Vector{Vector{S}}) where {F<:AbstractFloat, S<:AbstractState}
    for (n, state) in enumerate(ACEConfig.(states))
        A = evaluate(basis.basis, state)
        B = evaluateval_real_new(A)
        # B = evaluateval_real(A)        
        values[n, :, :] = (basis.coefficients' * B) + basis.mean
    end
end

# Fill a matrix with the results of a single state
function predict!(values::Matrix{F}, basis::Basis, states::Vector{S}) where {F<:AbstractFloat, S<:AbstractState}
    A = evaluate(basis.basis, ACEConfig(states))
    B = evaluateval_real_new(A)
    # B = evaluateval_real(A)  
    values .= (basis.coefficients' * B) + basis.mean
end

# Fill a matrix with the results of a single state
function predict!(values::AbstractArray{F, 2}, basis::Basis, state::Vector{S}) where {F<:AbstractFloat, S<:AbstractState}
    A = evaluate(basis.basis, ACEConfig(state))
    B = evaluateval_real_new(A)
    # B = evaluateval_real(A)
    values .= (basis.coefficients' * B) + basis.mean
end


# Construct and fill a matrix with the results of a single state
function predict(basis::Basis, states::Vector{S}) where S<:AbstractState
    n, m, type = ACE.valtype(basis.basis).parameters[3:5]
    values = Matrix{real(type)}(undef, n, m)
    predict!(values, basis, states)
    return values
end

# Fill a vector of matrices with the results from multiple states.
# This is mostly included only for compatibility i think and should be removed.
function predict!(values::Vector{Matrix{F}}, basis::Basis, states::Vector{Vector{S}}) where {F<:AbstractFloat, S<:AbstractState}
    for (n, state) in enumerate(ACEConfig.(states))
        A = evaluate(basis.basis, state)
        B = evaluateval_real_new(A)
        # B = evaluateval_real(A)
        values[n][:, :] .= (basis.coefficients' * B) + basis.mean
    end
end

# Fill multiple arrays with the results from multiple states
# This is used when filling sub-arrays; this is an important function and should be
# completely rewritten at a latter date.
function predict!(values::Vector{Any}, basis::Basis, states::Vector{Vector{S}}) where {S<:AbstractState}
    for (n, state) in enumerate(ACEConfig.(states))
        A = evaluate(basis.basis, state)
        B = evaluateval_real_new(A)
        # B = evaluateval_real(A)
        values[n][:, :] .= (basis.coefficients' * B) + basis.mean
    end
end




# Construct and fill a matrix with the results from multiple states
function predict(basis::Basis, states::Vector{Vector{S}}) where S<:AbstractState
    # Create a results matrix to hold the predicted values. The shape & type information
    # is extracted from the basis. However, complex types will be converted to their real
    # equivalents as results in ACEhamiltonians are always
    n, m, type = ACE.valtype(basis.basis).parameters[3:5]
    values = Array{real(type), 3}(undef, length(states), n, m)
    predict!(values, basis, states)
    return values
end

"""
# Todo
- Remove hardcoded matrix type.
- The indices of i & j will need to be swapped if zᵢ > zⱼ and the matrix transposed.
- Currently this only works for off-site blocks and thus requires refactorisation.
- Not sure how well this would work with complex numbers.
"""
# function predict_3(model::Model, atoms::Atoms, i::I, j::I) where I<:Integer
#     basis_def = model.basis_definition
#     zₛ = getfield.(atoms.Z, :z)
#     zᵢ, zⱼ = zₛ[i], zₛ[j]
#     noᵢ, noⱼ = number_of_orbitals(zᵢ, basis_def), number_of_orbitals(zⱼ, basis_def)
#     @assert zᵢ == zⱼ "Hetro-atomic blocks are not currently supported"
#     @assert i != j "On-site blocks are not currently supported"

#     matrix = Matrix{Float64}(undef, noᵢ, noⱼ)

#     matrix .= 0.0 # Debugging

#     shellsᵢ = basis_def[zᵢ]
#     shellsⱼ = basis_def[zⱼ]
#     n_shellsᵢ = length(shellsᵢ)
#     n_shellsⱼ = length(shellsⱼ)

#     # Get the bond state
#     current_envelope = CylindricalBondEnvelope(18.0,10.0,10.0) ###########
#     # current_envelope = nothing
#     bond_state = nothing
#     bond_state_r = nothing

#     # Will be replaced with something more efficient later on
#     n_orbsᵢ = shellsᵢ * 2 .+ 1
#     n_orbsⱼ = shellsⱼ * 2 .+ 1

#     sub_blocksᵢ = UnitRange{Int64}[i-j+1:i for (i, j) in zip(cumsum(n_orbsᵢ), n_orbsᵢ)]
#     sub_blocksⱼ = UnitRange{Int64}[i-j+1:i for (i, j) in zip(cumsum(n_orbsⱼ), n_orbsⱼ)]
#     for sᵢ in 1:n_shellsᵢ, sⱼ in 1:n_shellsⱼ

#         zᵢ == zⱼ && sᵢ > sⱼ && continue

#         basis = model.off_site_models[(zᵢ, zⱼ, sᵢ, sⱼ)]

#         # Check if the bond states need to be updated
#         if envelope(basis) != current_envelope
#             # current_envelope = envelope(basis)
#             bond_state = get_states(i, j, atoms, current_envelope)
#             bond_state_r = reverse.(bond_state)
#         end
        
#         @views sub_block = matrix[sub_blocksᵢ[sᵢ], sub_blocksⱼ[sⱼ]]
#         predict!(sub_block, basis, bond_state)

#         if zᵢ == zⱼ && sᵢ != sⱼ
#             # Parity induced sign flipping is not required as its effects are
#             # accounted for by the reversed bond state. 
#             @views sub_block = matrix[sub_blocksⱼ[sⱼ], sub_blocksᵢ[sᵢ]]'
#             predict!(sub_block, basis, bond_state_r)
#         end

#     end
#     return matrix
# end

"""
# Todo
- Consider cacheing state data as it will be more expensive to keep recalculating it.
  However, there will be a memory trade off and a larger up-front cost which might not
  be worth it if a cutoff distance is implemented; i.e. generating bond-states for all
  atom pairs when dealing with systems with thousands of atoms will be a bad idea as
  only a fraction would ever be in bonding distance. Might be worth using a cached
  version of the function.
- Might need to specify a r value for the off-site get_states function 
"""
function predict(model::Model, atoms::Atoms)

    # Currently this does not assign on-site data

    basis_def = model.basis_definition

    # Construct the matrix into which the data should be placed
    n_orbs = number_of_orbitals(atoms, basis_def)
    
    matrix = Matrix{Float64}(undef, n_orbs, n_orbs) # ← remove hardcoded type

    # Get a sorted list of all unique species present in the target system 
    atomic_numbers = getfield.(atoms.Z, :z)
    zₛ = sort(unique(atomic_numbers))

    # Loop over all unique species pairs then over all combinations of their shells.
    for (zₙ, zᵢ) in enumerate(zₛ)
        n_shellsᵢ = length(basis_def[zᵢ])

        # Might need to be regenerated when considering different on-site bases
        on_site_idxs = findall(atomic_numbers .== zᵢ)
        atomic_states = get_states.(on_site_idxs, (atoms,))
        
        for zⱼ in zₛ[zₙ:end]
            n_shellsⱼ = length(basis_def[zⱼ])

            # Identify atomic blocks & remove homo-atomic duplicates
            blocks_idxs = off_site_blocks(locate_blocks(zᵢ, zⱼ, atoms))
            blocks_idxs = zᵢ == zⱼ ? upper_blocks(blocks_idxs) : blocks_idxs

            # Todo: it would be best to ensure that bases with similar envelopes are 
            # looped over one after the other to prevent having to regenerated similar
            # states. Currently an arbitrary envolope 

            # Generate the bond ACE states
            current_envelope = nothing
            bond_states = nothing
            # envelope(basis)
            # get_states(blocks_idxs[:, 1], blocks_idxs[:, 2], (atoms,), (current_envelope,))
            
            # Loop over 
            for sᵢ in 1:n_shellsᵢ, sⱼ in 1:n_shellsⱼ

                # Skip symmetrically equivalent interactions.
                zᵢ == zⱼ && sᵢ > sⱼ && continue

                if zᵢ == zⱼ
                    on_site_basis = model.on_site_bases[(zᵢ, sᵢ, sⱼ)]
                    on_site_block_idxs = [on_site_idxs;; on_site_idxs] 
                    on_site_blocks = gather_matrix_blocks(matrix, on_site_block_idxs, sᵢ, sⱼ, atoms, basis_def)
                    predict!(on_site_blocks, on_site_basis, atomic_states)
                    # Assign data to the symmetrically equivalent sub-blocks
                    r_blocks = gather_matrix_blocks(matrix, on_site_block_idxs, sⱼ, sᵢ, atoms, basis_def)
                    for (b, rb) in zip(on_site_blocks, r_blocks)
                        rb .= b'
                    end
                end

                # Get the basis for the off site interaction 
                off_site_basis = model.off_site_bases[(zᵢ, zⱼ, sᵢ, sⱼ)]

                # Check if the bond states need to be updated
                if envelope(off_site_basis) != current_envelope
                    current_envelope = envelope(off_site_basis)
                    bond_states = get_states.(blocks_idxs[:, 1], blocks_idxs[:, 2], (atoms,), (current_envelope,))
                end

                # Gather the relevant off site sub-blocks matrices
                blocks = gather_matrix_blocks(matrix, blocks_idxs, sᵢ, sⱼ, atoms, basis_def)
                
                # Predict the sub-block and insert the values into the matrix
                predict!(blocks, off_site_basis, bond_states)
                
                # Assign data to the symmetrically equivalent sub-blocks
                r_blocks = gather_matrix_blocks(matrix, blocks_idxs[:, [2, 1]], sⱼ, sᵢ, atoms, basis_def)
                for (b, rb) in zip(blocks, r_blocks)
                    rb .= b'
                end
            end

        end
    end
    return matrix
end

end


