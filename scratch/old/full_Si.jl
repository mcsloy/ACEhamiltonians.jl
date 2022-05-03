using ACE, ACEbase, ACEhamiltonians, HDF5
using ACEbase: read_dict, write_dict, load_json, save_json
using ACEhamiltonians.DataIO: load_hamiltonian_gamma, load_atoms, load_basis_set_definition
using ACEhamiltonians.Fitting2: fit!, predict
using ACEhamiltonians.Parameters: ParaDef
using ACEhamiltonians.Bases: Model
using ACEhamiltonians.DataManipulation: collect_data
using ACEhamiltonians.DataManipulation: get_states, locate_blocks, upper_blocks, gather_matrix_blocks
using JuLIP: Atoms

path = "/home/ajmhpc/Documents/Work/Projects/ACEtb/Data/Si/Other/GammaOnlySupercell/Si_Special.hdf5"

print("Loading Data...")
H, atoms, basis_def = HDF5.h5open(path, "r") do database
    g = database["0548"]
    load_hamiltonian_gamma(g), load_atoms(g), load_basis_set_definition(g)
end
println("Done")

# Model construction and fitting
model = nothing
if isfile("Si_model_fitted.json")
    print("Loading fitted model...")
    model = read_dict(load_json("Si_model_fitted.json"))
    println("Done")
else
    if isfile("Si_model_unfitted.json")
        print("Loading unfitted model...")
        model = read_dict(load_json("Si_model_unfitted.json"))
        println("Done")
    else
        print("Building model...")
        model = Model(basis_def, ParaDef(basis_def, 2, 4, site="on"), ParaDef(basis_def, 2, 4, site="off"))
        println("Done")
        print("Saving unfitted model...")
        save_json("Si_model_unfitted.json", write_dict(model))
        println("Done")

    end
    print("Fitting model...")
    let database = HDF5.h5open(path, "r")
        fit!(model, [database["0548"]], :H)
    end
    println("Done")
    print("Saving fitted model...")
    save_json("Si_model_fitted.json", write_dict(model))
    println("Done")
end


# print("Predicting matrix...")
# results = predict(model, atoms)
println("Done")

# It looks like some, but not necessarily all, of the data is making it into the results
# matrix. However, there are some errors. There is also a maximum error of ~0.593.
# Looping over each off-site model and re-predicting the results shows that there is a
# non-trivial deviation between some of the models. This may be due to the limited 
# flexibility of the model or may be due to an error in how the data is collected.

# for (id, basis) in model.off_site_models
#     data_set = collect_data(H, basis, atoms, basis_def)
#     results = predict(basis, data_set.states)
#     delta = maximum(abs.(results - data_set.values))
#     println(id,": ", delta)
# end

# for (id, basis) in model.on_site_models
#     data_set = collect_data(H, basis, atoms, basis_def)
#     results = predict(basis, data_set.states)
#     delta = maximum(abs.(results - data_set.values))
#     println(id,": ", delta)
# end

# (14, 14, 2, 4): 1.3861003694454785e-6
# (14, 14, 2, 6): 0.0049729831900257315
# (14, 14, 1, 3): 0.0003822007254268931
# (14, 14, 6, 7): 0.04664456945270639
# (14, 14, 3, 3): 0.002278843690566804
# (14, 14, 4, 5): 0.001368103337847499
# (14, 14, 6, 6): 0.14005064086661095
# (14, 14, 1, 5): 0.0028501418464086936
# (14, 14, 5, 5): 0.030546876507666118
# (14, 14, 1, 2): 1.645819097930367e-7
# (14, 14, 1, 1): 0.0
# (14, 14, 2, 3): 0.00020499378877325385
# (14, 14, 3, 5): 0.008069197700089764
# (14, 14, 4, 7): 0.00039090969669344777
# (14, 14, 7, 7): 0.02076204727590372
# (14, 14, 1, 7): 0.0009868165315039242
# (14, 14, 4, 4): 6.1261704966101725e-6
# (14, 14, 5, 7): 0.026198210263852037
# (14, 14, 1, 4): 5.355691558620013e-7
# (14, 14, 2, 5): 0.0013364379788034612
# (14, 14, 3, 7): 0.010718483452450016
# (14, 14, 4, 6): 0.005560521873662334
# (14, 14, 1, 6): 0.010278708407656705
# (14, 14, 2, 2): 1.0155877763159168e-6
# (14, 14, 3, 4): 0.00010808538941969151
# (14, 14, 5, 6): 0.06697350197402281
# (14, 14, 3, 6): 0.011142492701445225
# (14, 14, 2, 7): 0.0005988669533593102

# (14, 1, 5): 9.202784657100404e-6
# (14, 1, 3): 6.820012471586581e-7
# (14, 2, 4): 0.00022988006037456566
# (14, 2, 7): 1.8646557580242158e-5
# (14, 3, 4): 2.9641952911573375e-5
# (14, 5, 6): 0.000545175895289518
# (14, 2, 5): 1.71841099362364e-5
# (14, 3, 7): 0.00019667107379143958
# (14, 4, 4): 0.0018606060789938539
# (14, 4, 7): 6.12541345087165e-5
# (14, 2, 3): 6.710235572722836e-6
# (14, 1, 2): 2.66491595106097e-6
# (14, 3, 5): 0.00027922902449086656
# (14, 6, 6): 0.0011111170913636337
# (14, 4, 5): 1.2538887197448192e-5
# (14, 3, 3): 0.0016475106921229887
# (14, 1, 6): 9.487536137791815e-6
# (14, 2, 2): 0.0018430168724679064
# (14, 5, 7): 0.00036550349449043296
# (14, 5, 5): 0.0014977786001111881
# (14, 6, 7): 0.0009846091701420996
# (14, 2, 6): 1.932112420812288e-5
# (14, 3, 6): 0.0001509902518189453
# (14, 1, 4): 4.430582855638202e-5
# (14, 4, 6): 1.1304953185616983e-5
# (14, 1, 7): 1.3265684543254507e-7
# (14, 7, 7): 0.0022566987202439215
# (14, 1, 1): 0.001844090485079164