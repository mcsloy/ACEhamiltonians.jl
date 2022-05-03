using ACE, ACEbase, ACEhamiltonians, HDF5, Statistics, BlockArrays, Serialization
using ACEbase: read_dict, write_dict, load_json, save_json
using ACEhamiltonians.DataIO: load_old_hamiltonian, load_old_atoms
using ACEhamiltonians.Fitting2: fit!, predict
using ACEhamiltonians.Predict: predict_offsite_HS
using ACEhamiltonians.Parameters: ParaDef
using ACEhamiltonians.Bases: Model, Basis, off_site_ace_basis, on_site_ace_basis
using ACEhamiltonians.DataManipulation: get_states, locate_blocks, upper_blocks, gather_matrix_blocks, collect_data, DataSet
using JuLIP: Atoms
using ACEhamiltonians.Predict: wmodels2err


function old_parameters()
    r_cut = repeat([12.], 9)
    max_deg = repeat([8], 9)
    order = repeat([2], 9)
    λ = repeat([1e-7], 9)
    reg_type = 2
    return Params(r_cut, max_deg, order, λ, reg_type, "LSQR")
end




path = "/home/ajmhpc/Projects/ACEtb/Data/test_data/FCC-MD-500K/SK-supercell-001.h5"
n = 11
allowed_blocks = [(i,j) for i=1:n for j=1:n if i!=j]

print("Loading matrix data...")
H, atoms = load_old_hamiltonian(path), load_old_atoms(path)
println("Done")



old_model = nothing
if isfile("Al_old_fitted_off.bin")
    print("Loading old fitted model from binary...")
    old_model = deserialize("Al_old_fitted_off.bin")
    println("Done")
elseif isfile("Al_old_fitted_off.json")
    print("Loading old fitted model...")
    old_model = read_dict(load_json("Al_old_fitted_off.json"))
    println("Done")
else

    print("Constructing fitted model...")
    data_info = Data([path,], allowed_blocks)
    old_model, _, _ = params2wmodels(data_info, old_parameters(), old_parameters())
    println("Done")

    print("Saving old fitted model...")
    save_json("Al_old_fitted_off.json", write_dict(old_model))
    println("Done")
end


basis_def = Dict(13=>[0, 0, 0, 1, 1, 2])
if isfile("Al_model_fitted.bin")
    print("Loading fitted model from binary...")
    model = deserialize("Al_model_fitted.bin")
    println("Done")
elseif isfile("Al_model_fitted.json")
    print("Loading fitted model...")
    model = read_dict(load_json("Al_model_fitted.json"))
    println("Done")
else
    if isfile("Al_model_unfitted.bin")
        print("Loading unfitted model from binary...")
        model = deserialize("Al_model_unfitted.bin")
        println("Done")
    elseif isfile("Al_model_unfitted.json")
        print("Loading unfitted model...")
        model = read_dict(load_json("Al_model_unfitted.json"))
        println("Done")
    else
        print("Building model...")
        model = Model(basis_def, ParaDef(basis_def, 2, 8, site="on"), ParaDef(basis_def, 2, 8, site="off"))
        println("Done")
        print("Saving unfitted model...")
        serialize("Al_model_unfitted.json", model)
        # save_json("Al_model_unfitted.json", write_dict(model))
        println("Done")
    end

    println("Fitting on-site models")
    allowed_atoms = collect(1:n)
    for (id, basis) in model.on_site_models
        print("Fitting: $id...")
        data_set = collect_data(H, basis, atoms, basis_def, allowed_atoms)
        fit!(basis, data_set)
        println("Done")
    end

    println("Fitting off-site models")
    for (id, basis) in model.off_site_models
        print("Fitting: $id...")
        data_set = collect_data(H, basis, atoms, basis_def, allowed_atoms)
        fit!(basis, data_set)
        println("Done")
    end

    print("Saving fitted model...")
    serialize("Al_model_fitted.json", model)
    # save_json("Al_model_fitted.json", write_dict(model))
    println("Done")
    #serialize("Al_old_model_fitted.bin", old_model)

end


# Need to walk-through the calculation of values step by step and identify where
# the deviation in the upper triangle originates from. During testing it will be
# worth swapping the bases around; i.e copying over the basis functions from the
# old model to the newer one to help remove variables.
# Need to use CylindricalBondEnvelope(18.0,10.0,10.0) to ensure consistency.

# It also might be worth re-fitting the models on only the data from a single block
# this may help eliminate variance deviations between to old and new models.

# sp = model.off_site_models[(13, 13, 1, 4)]
# state_12 = get_states(1, 2, atoms, envelope(sp))
# state_21 = get_states(1, 2, atoms, envelope(sp))



# block_12 = H[a2b(1), a2b(2)]
# block_21 = H[a2b(2), a2b(1)]
# block_12_s1_p4 = block_12[1:1, 4:6]
# block_12_p4_s1 = block_12[4:6, 1:1]

# block_21_s1_p4 = block_21[1:1, 4:6]
# block_21_p4_s1 = block_21[4:6, 1:1]

#[1, 2, 3, 4, 7, 10] # starts
#[1, 2, 3, 6, 9, 14] # ends


# Atomic index to block slice
# a2b(i) = (1:14) .+ ((i - 1) * 14)
# stack(i) = collect(reduce(vcat, i))

# H_p = predict(model, atoms)
# blocks_ref = stack([reshape(H[a2b.(i)...], 1, 14, 14) for i in allowed_blocks])
# blocks_new = stack([reshape(H_p[a2b.(i)...], 1, 14, 14) for i in allowed_blocks])
# blocks_old = permutedims(predict_offsite_HS(atoms, old_model, allowed_blocks), (3, 1, 2))