function to_dataset(mat, states, i, j)
    c, r = size(mat[1][i,j])
    return DataSet(
        collect(reduce(vcat, reshape.((m->m[i,j]).(mat), 1, c, r))),
        ones(Int64, length(states), 2),
        ones(Int64, length(states)),
        states)
end

atoms, basis_def, images, S = HDF5.h5open(path, "r") do database
    g = database["0244"]
    load_atoms(g), load_basis_set_definition(g), load_cell_translations(g), load_overlap(g)
end