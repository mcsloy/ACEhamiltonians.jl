using ACE, ACEbase, ACEhamiltonians, HDF5, Statistics, BlockArrays, Serialization
using ACE: CylindricalBondEnvelope, ACEConfig, evaluate, scaling, AbstractState
using ACEbase: read_dict, write_dict, load_json, save_json
using ACEhamiltonians.DataIO: load_old_hamiltonian, load_old_atoms
using ACEhamiltonians.Fitting2: fit!, predict, predict!
using ACEhamiltonians.Predict: predict_offsite_HS
using ACEhamiltonians.Parameters: ParaDef
using ACEhamiltonians.Bases: Model, Basis, off_site_ace_basis, on_site_ace_basis
using ACEhamiltonians.DataManipulation: get_states, locate_blocks, upper_blocks, gather_matrix_blocks, collect_data, DataSet, collect_data_r
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

        basis = model.off_site_bases[(zᵢ, zⱼ, sᵢ, sⱼ)]

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

_shells = [0, 0, 0, 1, 1, 2]
_n_shells = _shells * 2 .+ 1
_shell_ends = cumsum(_n_shells)
_shell_starts = _shell_ends - _n_shells .+ 1

a2b(i) = (1:14) .+ ((i - 1) * 14)
b2s(i) = _shell_starts[i]:_shell_ends[i]

env = CylindricalBondEnvelope(18.0,10.0,10.0)
path = "/home/ajmhpc/Projects/ACEtb/Data/test_data/FCC-MD-500K/SK-supercell-001.h5"
n = 30
allowed_blocks = [(i,j) for i=1:n for j=1:n if i!=j]

H, atoms = load_old_hamiltonian(path), load_old_atoms(path)

block_12 = H[a2b(1), a2b(2)]
block_21 = H[a2b(2), a2b(1)]

# old_model = deserialize("Al_old_model_fitted.bin");
model = deserialize("Al_model_fitted.bin");
basis_def = model.basis_definition
basis_dd = model.off_site_bases[(13, 13, 6, 6)]

print("Collecting Data...")
data_set_upper = collect_data(H, basis_dd, atoms, basis_def, collect(1:n))
data_set_lower = collect_data_r(H, basis_dd, atoms, basis_def, collect(1:n))
data_set_full = data_set_upper + data_set_lower
n_states = length(data_set_upper)
println("Done")

reference_upper = zeros(5, 5, n_states)
reference_lower = zeros(5, 5, n_states)
for i in 1:n_states
    zᵢ, zⱼ = data_set_upper.block_indices[i, :]
    reference_upper[:, :, i] = H[a2b(zᵢ), a2b(zⱼ)][b2s(6), b2s(6)]
    reference_lower[:, :, i] = H[a2b(zⱼ), a2b(zᵢ)][b2s(6), b2s(6)]
end

# Upper only
upper_only_coefficients = basis_dd.coefficients
upper_only_values_u = zeros(5, 5, n_states)
upper_only_values_l = zeros(5, 5, n_states)
print("Predicting...")
for i in 1:n_states
    upper_only_values_u[:, :, i] = predict(basis_dd, data_set_upper.states[i])
    upper_only_values_l[:, :, i] = predict(basis_dd, reverse.(data_set_upper.states[i]))
end
println("Done")

# Lower only
print("Fitting...")
fit!(basis_dd, data_set_lower)
println("Done")
lower_only_coefficients = basis_dd.coefficients
lower_only_values_u = zeros(5, 5, n_states)
lower_only_values_l = zeros(5, 5, n_states)
print("Predicting...")
for i in 1:n_states
    lower_only_values_u[:, :, i] = predict(basis_dd, data_set_upper.states[i])
    lower_only_values_l[:, :, i] = predict(basis_dd, reverse.(data_set_upper.states[i]))
end
println("Done")

# Full
print("Fitting...")
fit!(basis_dd, data_set_full)
println("Done")
full_coefficients = basis_dd.coefficients
full_values_u = zeros(5, 5, n_states)
full_values_l = zeros(5, 5, n_states)
print("Predicting...")
for i in 1:n_states
    full_values_u[:, :, i] = predict(basis_dd, data_set_upper.states[i])
    full_values_l[:, :, i] = predict(basis_dd, reverse.(data_set_upper.states[i]))
end
println("Done")


error_t(i, j) = mean(abs.(i - permutedims(j, (2, 1, 3))), dims=3)

error_t_u = mean(error_t(upper_only_values_u, upper_only_values_l))
error_t_l = mean(error_t(lower_only_values_u, lower_only_values_l))
error_t_f = mean(error_t(full_values_u, full_values_l))

error_uu = mean(abs.(upper_only_values_u - reference_upper))
error_ul = mean(abs.(upper_only_values_l - reference_lower))

error_lu = mean(abs.(lower_only_values_u - reference_upper))
error_ll = mean(abs.(lower_only_values_l - reference_lower))

error_fu = mean(abs.(full_values_u - reference_upper))
error_fl = mean(abs.(full_values_l - reference_lower))



# 
#


# data_set = collect_data(H, basis_dd, atoms, basis_def, collect(1:n))
# A_1 = evaluate(basis_dd.basis, ACEConfig(state_12))
# # B = evaluateval_real_new(A)
# B_1 = evaluateval_real(A_1);
# values_1 = basis_dd.coefficients' * B_1

# A_2 = evaluate(basis_dd.basis, ACEConfig(state_21))
# # B = evaluateval_real_new(A)
# B_2 = evaluateval_real(A_2)
# values_2 = basis_dd.coefficients' * B_2


# bb_basis = model.off_site_models[(13,13,6,6)]
# old_dd = old_model.ModelDD[1];
# new_dd = TBModel(dd_basis.basis, dd_basis.coefficients, dd_basis.mean);

# old_dd_12_original = predict_offsite_HS(atoms, old_model, [(1, 2)])[10:14, 10:14, 1]
# new_dd_12 = predict_3(model, atoms, 1, 2)[10:14, 10:14]
# old_model.ModelDD[1] = new_dd;
# old_dd_12_updated = predict_offsite_HS(atoms, old_model, [(1, 2)])[10:14, 10:14, 1]

# pp_basis = model.off_site_models[(13,13,5,5)];
# old_pp = old_model.ModelPP[4];
# new_pp = TBModel(pp_basis.basis, pp_basis.coefficients, pp_basis.mean);

# old_pp_12_original = predict_offsite_HS(atoms, old_model, [(1, 2)])[7:9, 7:9, 1];
# new_pp_12 = predict_3(model, atoms, 1, 2)[7:9, 7:9];
# old_model.ModelPP[4] = new_pp;
# old_pp_12_updated = predict_offsite_HS(atoms, old_model, [(1, 2)])[7:9, 7:9, 1];


# sp_basis = model.off_site_models[(13,13,1,4)];
# old_sp = old_model.ModelSP[1];
# new_sp = TBModel(sp_basis.basis, sp_basis.coefficients, sp_basis.mean);

# old_sp_12_original = predict_offsite_HS(atoms, old_model, [(1, 2)])[1:1, 4:6, 1];
# new_sp_12 = predict_3(model, atoms, 1, 2)[1:1, 4:6];
# old_model.ModelSP[1] = new_sp;
# old_sp_12_updated = predict_offsite_HS(atoms, old_model, [(1, 2)])[1:1, 4:6, 1];

# old_mat = predict_offsite_HS(atoms, old_model, [(1, 2)])[:, :, 1];

# n = 0
# for j=4:5, i=1:3
#     n += 1;
#     i > j && continue
#     basis = model.off_site_models[(13, 13, i, j)];
#     old_model.ModelSP[n] = TBModel(basis.basis, basis.coefficients, basis.mean);
# end

# updated_mat = predict_offsite_HS(atoms, old_model, [(1, 2)])[:, :, 1];

# new_mat = predict_3(model, atoms, 1, 2)
# 1 4
# 1 5
# 2 4
# 2 5
# 3 4
# 4 5

# abs.(predict_offsite_HS(atoms, old_model, [(1, 2)])[:, :, 1]- predict_offsite_HS(atoms, old_model, [(2, 1)])[:, :, 1]')