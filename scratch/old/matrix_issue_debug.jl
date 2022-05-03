using ACE, ACEbase, ACEhamiltonians, HDF5, Statistics, BlockArrays
using ACEbase: read_dict, write_dict, load_json, save_json
using ACEhamiltonians.DataIO: load_old_hamiltonian, load_old_atoms
using ACEhamiltonians.Fitting2: fit!, predict
using ACEhamiltonians.Parameters: ParaDef
using ACEhamiltonians.Bases: Model, Basis, off_site_ace_basis, on_site_ace_basis
using ACEhamiltonians.DataManipulation: get_states, locate_blocks, upper_blocks, gather_matrix_blocks, collect_data, DataSet
using JuLIP: Atoms
using ACEhamiltonians.Predict: wmodels2err


function old_parameters()
    r_cut = repeat([12.], 9)
    max_deg = repeat([4], 9)
    order = repeat([2], 9)
    λ = repeat([1e-7], 9)
    reg_type = 2
    return Params(r_cut, max_deg, order, λ, reg_type, "LSQR")
end


function get_new_model()
    model = nothing
    basis_def = Dict(13=>[0, 0, 0, 1, 1, 2])
    if isfile("Al_model_fitted.json")
        print("Loading fitted model...")
        model = read_dict(load_json("Al_model_fitted.json"))
        println("Done")
    else
        if isfile("Al_model_unfitted.json")
            print("Loading unfitted model...")
            model = read_dict(load_json("Al_model_unfitted.json"))
            println("Done")
        else
            print("Building model...")
            model = Model(basis_def, ParaDef(basis_def, 2, 4, site="on"), ParaDef(basis_def, 2, 4, site="off"))
            println("Done")
            print("Saving unfitted model...")
            save_json("Al_model_unfitted.json", write_dict(model))
            println("Done")
        end
    end
    return model
end




path = "/home/ajmhpc/Projects/ACEtb/Data/test_data/FCC-MD-500K/SK-supercell-001.h5"
allowed_blocks = [(i,j) for i=1:4 for j=1:4 if i!=j]

print("Loading matrix data...")
H, atoms = load_old_hamiltonian(path), load_old_atoms(path)
println("Done")


new_model = get_new_model()
old_model = nothing
if isfile("Al_old.json")
    print("Loading old model...")
    old_model = read_dict(load_json("Al_old.json"))
    println("Done")
else
    print("Constructing data info...")
    data_info = Data([path,], allowed_blocks)
    println("Done")

    print("Fitting model...")
    old_model, _, data_whole = params2wmodels(data_info, old_parameters(), old_parameters())
    println("Done")
end

old_blocks = permutedims(predict_offsite_HS(atoms, old_model, allowed_blocks), (3, 1, 2))
ref_blocks = collect(reduce(vcat, [reshape(H[(1:14) .+ ((i[1] - 1) * 14), (1:14) .+ ((i[2] - 1) * 14)], 1, 14, 14)  for i in allowed_blocks]))
_block_idxs = [1:1, 2:2, 3:3, 4:6, 7:9, 10:14]
get_old_blocks(i, j) = old_blocks[:, _block_idxs[i], _block_idxs[j]]
get_ref_blocks(i, j) = H[:, _block_idxs[i], _block_idxs[j]]





# data_set_11 = to_dataset(data_whole[1][1], data_whole[1][3], 1, 1)
# basis_11 = new_model.off_site_models[(13, 13, 1, 1)]
# fit!(basis_11, data_set_11)
# new_results_11 = predict(basis_11, data_set_11.states)
# delta_11 = mean(abs.(new_results_11 - get_old_blocks(1, 1)))


# data_set_14 = to_dataset(data_whole[2][1], data_whole[2][3], 1, 1)
# basis_14 = new_model.off_site_models[(13, 13, 1, 4)]
# fit!(basis_14, data_set_14)
# new_results_14 = predict(basis_14, data_set_14.states)
# delta_14 = mean(abs.(new_results_14 - get_old_blocks(1, 4)))

# function get_delta(k, i, j, n, m)
#     data_set_x = to_dataset(data_whole[k][1], data_whole[k][3], i, j)
#     basis_x = new_model.off_site_models[(13, 13, n, m)]
#     fit!(basis_x, data_set_x)
#     new_results = predict(basis_x, data_set_x.states)
#     return mean(abs.(new_results - get_old_blocks(n, m)))
# end


# path = "/home/ajmhpc/Documents/Work/Projects/ACEtb/Data/Si/Other/GammaOnlySupercell/Si_Special.hdf5"

# print("Loading Data...")
# H_ref, atoms, basis_def = HDF5.h5open(path, "r") do database
#     g = database["0548"]
#     load_hamiltonian_gamma(g), load_atoms(g), load_basis_set_definition(g)
# end
# println("Done")

# basis = Basis(off_site_ace_basis(1, 1, 2, 8, 12.0), (14, 14, 6, 6))
# data_set_train = collect_data(H_ref, basis, atoms,  basis_def)
# fit!(basis, data_set_train)

# data_set_test_a = collect_data(H_ref, basis, atoms, basis_def)
# predicted = predict(basis, data_set_test_a.states)
# delta = maximum(abs.(predicted - data_set_test_a.values))

# delta = maximum(abs.(predicted - data_set_train.values))