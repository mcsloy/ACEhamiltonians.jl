using JuLIP


function build_environment(basis::Basis, atoms::JuLIP.Atoms)
end


# using BlockArrays

# 


# # Need a method by which to specify the target blocks
# # The base method will be manual selection of atom pairs
# # There will be higher level constructors which allow for a more curated selection process


# mutable struct Data
#     species
#     shells
#     blocks
#     sub_blocks
#     values
# end




# """
# Partition a matrix into its atom-atom blocks.
# """

# function atom_block_matrix(matrix, atomic_numbers, basis_def)
#     # Work out the number of orbitals present on each atom in the system
#     n_orbs_atoms = get.(
#         # Raise an error if a key is missing (response if key is absent)
#         (() -> error("Species definition is incomplete"),),
#         # Number of orbitals on each species (the dictionary to query)
#         (Dict(i=>sum(2 .* j .+ 1) for (i, j) in basis_def),),
#         # Atomic numbers of the species present (value to query)
#         atomic_numbers)
#     # Use this to partition the matrix into its atomic blocks
#     return BlockArray(matrix, n_orbs_atoms, n_orbs_atoms)
# end


# # Three implementations of this function are required:
# #   1) To deal with a single "standard" system
# #   2) To deal with a periodically reduced system.
# #   3) To deal with multiple systems.
# # TODO:
# #   - Need to add handling for on-site variant:
# #       - This should ignore the lower part of the on-site block as the data
# #         is a repeat of the upper part of the block.
# #   - Need special handling of homo-atomic off-site systems:
# #       - This should 1: combine symmetrically equivalent data. 


# function collect_data(matrix::Matrix, type, atomic_numbers, basis_def)

#     # TODO:
#     #   - Need special handling for on 
    



#     # Calculate the number of orbitals present on each shell and on each species
#     n_orbs_s = Dict(i=> 2 .* j .+ 1 for (i, j) in basis_def)
#     n_orbs_a = Dict(i=>sum(j) for (i, j) in n_orbs_s)

#     z₁, z₂ = type[1], type[end-2]
#     s₁, s₂ = type[end-1:end]
#     n₁, n₂ = n_orbs_s[z₁][s₁], n_orbs_s[z₂][s₂]

#     # Build a row/column slicer pair that can be used to gather the desired shell-shell
#     # sub-block from an appropriate atom block.
#     f = (zᵢ, sᵢ) -> (a -> a[sᵢ] - a[1]:1:a[sᵢ])(cumsum(n_orbs_s[zᵢ]))
#     sb_rows, sb_cols = f.((z₁, z₂), (s₁, s₂))

#     # Identify where each atomic-block starts
#     block_i = cumsum(getindex.(Ref(n_orbs_a), atomic_numbers))
#     block_i = block_i .- block_i[1]

#     # Gather the row and column start indices for the relevant atom-blocks. 
#     b_rows = block_i[findall(==(z₁), atomic_numbers)]
#     b_cols = block_i[findall(==(z₂), atomic_numbers)]


#     if length(target) == 3 # If targeting on-site blocks
#         # Then only blocks on the diagonal should be considered
#         idxs = ((r, c) for (r, c) in zip(b_rows, b_cols))
#         data = Array{eltype(matrix), 3}(length(b_rows), n₁, n₂)
#         for (n, (cᵢ, rᵢ)) in enumerate(idxs)
#             data[n, :, :] = matrix[rᵢ .+ sb_rows, cᵢ .+ sb_cols]
#         end


#     else # If targeting off-site blocks
#         if z₁ == z₂ # If interactions are homo-atomic
#             # The pull
#             if s₁ == s₂ # If interactions are between the same shell
#                 # Then only pull data from atomic-blocks in the upper half of the matrix
#                 idxs =((r, c) for r in b_rows for c in b_cols if r>c)
#             else # If interactions are between different shells
#                 # Then pull the upper triangle out 
#             end
#         else # If interactions are hetero-atomic
#             # Then look at all atomic-blocks other than those on the diagonal
#             idxs =((r, c) for r in b_rows for c in b_cols if r≠c)

#             data = Array{eltype(matrix), 3}(length(b_rows), n₁, n₂)
#             for (n, (cᵢ, rᵢ)) in enumerate(idxs)
#                 data[n, :, :] = matrix[rᵢ .+ sb_rows, cᵢ .+ sb_cols]
#             end
#         end
#     end

#     # Select the 



#     data = Array{eltype(matrix), 3}(undef, n_orbs_s[z₁][s₁], n_orbs_s[z₂][s₂])


    


#     n = 0
#     for b_row=b_rows, b_col=b_cols
#         n += 1
#         matrix[b_row, ]
#     end



    
    
        
        

# end



# basis_def = Dict(1=>[0], 6=>[0,1])
# mat = rand(8, 8)
# atoms = [6, 1, 1, 1, 1]

# b_mat = atom_block_matrix(mat, atoms, basis_def)
