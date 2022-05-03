using ACE, ACEbase, ACEhamiltonians, HDF5, Statistics, BlockArrays, Serialization
using ACE: CylindricalBondEnvelope
using ACEbase: read_dict, write_dict, load_json, save_json
using ACEhamiltonians.DataIO: load_old_hamiltonian, load_old_atoms
using ACEhamiltonians.Fitting2: fit!, predict, predict!
using ACEhamiltonians.Predict: predict_offsite_HS
using ACEhamiltonians.Parameters: ParaDef
using ACEhamiltonians.Bases: Model, Basis, off_site_ace_basis, on_site_ace_basis
using ACEhamiltonians.DataManipulation: get_states, locate_blocks, upper_blocks, gather_matrix_blocks, collect_data, DataSet
using JuLIP: Atoms
using ACEhamiltonians.Structure: TBModel
using ACEhamiltonians.Predict: wmodels2err


function predict_3(model::Model, atoms::Atoms, i::I, j::I) where I<:Integer
    basis_def = model.basis_definition
    zₛ = getfield.(atoms.Z, :z)
    zᵢ, zⱼ = zₛ[i], zₛ[j]
    noᵢ, noⱼ = number_of_orbitals(zᵢ, basis_def), number_of_orbitals(zⱼ, basis_def)
    @assert zᵢ == zⱼ "Hetro-atomic blocks are not currently supported"
    @assert i != j "On-site blocks are not currently supported"

    matrix = Matrix{Float64}(undef, noᵢ, noⱼ)

    matrix .= 0.0 # Debugging

    shellsᵢ = basis_def[zᵢ]
    shellsⱼ = basis_def[zⱼ]
    n_shellsᵢ = length(shellsᵢ)
    n_shellsⱼ = length(shellsⱼ)

    # Get the bond state
    current_envelope = CylindricalBondEnvelope(18.0,10.0,10.0) ###########
    # current_envelope = nothing
    bond_state = nothing
    bond_state_r = nothing

    # Will be replaced with something more efficient later on
    n_orbsᵢ = shellsᵢ * 2 .+ 1
    n_orbsⱼ = shellsⱼ * 2 .+ 1

    sub_blocksᵢ = UnitRange{Int64}[i-j+1:i for (i, j) in zip(cumsum(n_orbsᵢ), n_orbsᵢ)]
    sub_blocksⱼ = UnitRange{Int64}[i-j+1:i for (i, j) in zip(cumsum(n_orbsⱼ), n_orbsⱼ)]
    for sᵢ in 1:n_shellsᵢ, sⱼ in 1:n_shellsⱼ

        zᵢ == zⱼ && sᵢ > sⱼ && continue

        basis = model.off_site_models[(zᵢ, zⱼ, sᵢ, sⱼ)]

        # Check if the bond states need to be updated
        if envelope(basis) != current_envelope
            # current_envelope = envelope(basis)
            bond_state = get_states(i, j, atoms, current_envelope)
            bond_state_r = reverse.(bond_state)
        end
        
        @views sub_block = matrix[sub_blocksᵢ[sᵢ], sub_blocksⱼ[sⱼ]]
        predict!(sub_block, basis, bond_state)

        if zᵢ == zⱼ && sᵢ != sⱼ
            # Parity induced sign flipping is not required as its effects are
            # accounted for by the reversed bond state. 
            @views sub_block = matrix[sub_blocksⱼ[sⱼ], sub_blocksᵢ[sᵢ]]'
            predict!(sub_block, basis, bond_state_r)
        end

    end
    return matrix
end


# model = deserialize("Al_model_fitted.bin");

# path = "/home/ajmhpc/Projects/ACEtb/Data/test_data/FCC-MD-500K/SK-supercell-001.h5"
# n = 5
# allowed_blocks = [(i,j) for i=1:n for j=1:n if i!=j]

# H, atoms = load_old_hamiltonian(path), load_old_atoms(path)

# a2b(i) = (1:14) .+ ((i - 1) * 14)

# block_12 = H[a2b(1), a2b(2)]
# block_21 = H[a2b(2), a2b(1)]

# t1 = time()
# model_json = load_json("/home/ajmhpc/Projects/ACEhamiltonians/Code/Working/Basis/Al_model_fitted_old.json");
# t2 = time()
# println("Time to parse json: $(t2 - t1)")
# model = read_dict(model_json);
# t3 = time()
# println("Time to parse dict: $(t3 - t2)")
# serialize("/home/ajmhpc/Projects/ACEhamiltonians/Code/Working/Basis/Al_model_fitted.bin", model)
# println("Model has been dumped into binary")
# t1 = time()
# dict_out = write_dict(model)
# t2 = time()
# println("Time to create dict: $(t2 - t1)")
# save_json("/home/ajmhpc/Projects/ACEhamiltonians/Code/Working/Basis/Al_model_fitted.bin", dict_out)
# t3 = time()
# println("Time to save json: $(t3 - t2)")
# # old_model = deserialize("Al_old_model_fitted.json");
# # model = deserialize("Al_model_fitted.bin");
# # off_basis_11 = model.off_site_models[(13, 13, 1, 1)]
# # off_basis_44 = model.off_site_models[(13, 13, 4, 4)]
# # off_basis_22 = model.off_site_models[(13, 13, 2, 2)]
# # on_basis = model.on_site_models[(13, 1, 1)]