using Serialization, Statistics, LinearAlgebra, JuLIP, NeighbourLists
using JuLIP.Utils: project_min
import ACE: evaluate!
import ACEbase: evaluate!
using ACEhamiltonians.Bases: Basis, off_site_ace_basis, envelope
using ACEhamiltonians.DataIO: load_old_hamiltonian, load_old_atoms
using ACEhamiltonians.DataManipulation: collect_data, collect_data_r, get_states, DataSet, BondState
using ACE: CylindricalBondEnvelope, Rn1pBasis, Ylm1pBasis, _rr, _inner_evaluate
using ACEhamiltonians.Fitting2: fit!, predict
using ACE: cat2cyl, _eff_zcut, _eval_env_inner


path = "/home/ajmhpc/Projects/ACEtb/Data/test_data/FCC-MD-500K/SK-supercell-001.h5"
_shells = [0, 0, 0, 1, 1, 2]
_norbs = _shells * 2 .+ 1
_ends = cumsum(_norbs)
_starts = _ends - _norbs .+ 1

a2b(i) = (1:14) .+ ((i - 1) * 14)
b2s(i) = _starts[i]:_ends[i]
println("Collecting data")
H, atoms = load_old_hamiltonian(path), load_old_atoms(path)
basis_def = Dict(13=>[0, 0, 0, 1, 1, 2])
basis_0 = nothing
basis_1 = nothing
env_0 = nothing
env_1 = nothing

if isfile("basis_dd_0.bin")
    println("Loading model")
    basis_0 = deserialize("basis_dd_0.bin")
    env_0 = envelope(basis_0)
else
    println("Building model")
    env_0 = CylindricalBondEnvelope(18.0, 12.0, 5.0, λ=0.0, floppy=false)
    basis_0 = Basis(off_site_ace_basis(2, 2, 2, 8, env_0), (13, 13, 6, 6))
    serialize("basis_dd_0.bin", basis_0)
end

if isfile("basis_dd_1.bin")
    println("Loading model")
    basis_1 = deserialize("basis_dd_1.bin")
    env_1 = envelope(basis_1)
else
    println("Building model")
    env_1 = CylindricalBondEnvelope(18.0, 12.0, 5.0, floppy=false)
    basis_1 = Basis(off_site_ace_basis(2, 2, 2, 8, env_1), (13, 13, 6, 6))
    serialize("basis_dd_1.bin", basis_1)
end


function check(basis, data_set_1, data_set_2)
    fit!(basis, data_set_1)
    a = predict(basis_0, data_set_1.states[1])
    b = predict(basis_0, data_set_2.states[1])
    return mean(abs.(a - b'))
end

function state_mod_1(data_set)
    return DataSet(
        data_set.values, data_set.block_indices, data_set.cell_indices,
        [[BondState(x[1].rr/norm(x[1].rr), x[1].rr0, :bond); x[2:end]] for x in data_set.states]
        )
end


function state_mod_2(data_set)
    return DataSet(
        data_set.values, data_set.block_indices, data_set.cell_indices,
        [[BondState(typeof(x[1].rr)(ones(3)) .* sign.(x[1].rr), x[1].rr0, :bond); x[2:end]] for x in data_set.states]
        )
end

basis_2 = Basis(off_site_ace_basis(2, 2, 2, 8, env_0, 0.1), (13, 13, 6, 6));


ds_12_0 = collect_data(H, basis_0, atoms, basis_def, [1, 2])
ds_21_0 = collect_data_r(H, basis_0, atoms, basis_def, [1, 2])

ds_12_1 = collect_data(H, basis_1, atoms, basis_def, [1, 2])
ds_21_1 = collect_data_r(H, basis_1, atoms, basis_def, [1, 2])

# 1.2381293951503337e-5
# 0.09381759238489523
# 0.09381798432752378

ds_12_0s = state_mod_1(ds_12_0)
ds_21_0s = state_mod_1(ds_21_0)
# R+Y
# u,l,t
# 0.0542
# 0.1859
# 0.1671

# R
# 0.00320
# 0.08231
# 0.08153

# Y
# 0.001195373814112361
# 0.17139966519843164
# 0.1711365717078394)

# None
# 8.379933154037956e-5
# 0.0855846534611728
# 0.08555879884863994)

fit!(basis_2, ds_12_0)
(mean(abs.(predict(basis_2, ds_12_0.states[1]) - H[a2b(1), a2b(2)][b2s(6), b2s(6)])),
mean(abs.(predict(basis_2, ds_21_0.states[1]) - H[a2b(2), a2b(1)][b2s(6), b2s(6)])),
mean(abs.(predict(basis_2, ds_12_0.states[1]) - predict(basis_2, ds_21_0.states[1])')))

fit!(basis_0, ds_12_0)
fit!(basis_2, ds_12_0)

# aᵢ, aⱼ = 1, 2

# pair_list = JuLIP.neighbourlist(atoms, 20.0)
# rr0 = project_min(atoms, atoms.X[aⱼ] - atoms.X[aᵢ])

# offset = rr0 / 2

# idxs, vecs = NeighbourLists.neigs(pair_list, aᵢ)

# env_states_o = BondState{typeof(rr0)}[BondState(v, rr0, :env) for v in vecs[1:end .!= j_idx]]
# env_states_n = BondState{typeof(rr0)}[BondState(v - offset, rr0, :env) for v in vecs[1:end .!= j_idx]]

# TODO work out exactly how the environment is being culled.

# 4.514173104207323

# 0.04662982961594905
# 0.25235263077596526
# println("Collecting data")
# data_set_1_2 = collect_data(H, basis_0, atoms, basis_def, [1, 2])
# data_set_2_1 = collect_data_r(H, basis_0, atoms, basis_def, [1, 2])


# println("Collecting data")
# data_set = collect_data(H, basis, atoms, basis_def, [1, 2, 3, 4])
# println("Fitting")
# fit!(basis, data_set)
# f1(i) = BondState(-(i.rr/2), i.rr0, :bond)
# f2(i) = BondState((i.rr/2), i.rr0, :bond)
# f3(i) = BondState(-i.rr, i.rr0, :bond)

# shift(d, func) = DataSet(
#     d.values, d.block_indices, d.cell_indices,
#     [[func(i[1]); i[2:end]] for i in d.states])


# data_set_a = collect_data(H, basis_0, atoms, basis_def, [1, 2, 3, 4, 5, 6, 7, 8])
# data_set_b = collect_data(H, basis_1, atoms, basis_def, [1, 2, 3, 4, 5, 6, 7, 8])


# fit!(basis_0, data_set_a)
# fit!(basis_1, data_set_b)


# e1 = mean(abs.(predict(basis_0, ds_12_0.states[1]) - predict(basis_0, ds_21_0.states[1])'))

# e2 = mean(abs.(predict(basis_1, ds_12_1.states[1]) - predict(basis_1, ds_21_1.states[1])'))
# data_set_a = collect_data(H, basis, atoms, basis_def, [1, 2, 3, 4, 5, 6, 7, 8])

# e_new = Float64[]
# e_old = Float64[]

# for i=1:5, j=1:5
#     i >= j && continue

#     xds_12_0 = collect_data(H, basis_0, atoms, basis_def, [i, j])
#     xds_21_0 = collect_data_r(H, basis_0, atoms, basis_def, [i, j])

#     xds_12_1 = collect_data(H, basis_1, atoms, basis_def, [i, j])
#     xds_21_1 = collect_data_r(H, basis_1, atoms, basis_def, [i, j])

#     append!(e_new, check(basis_0, xds_12_0, xds_21_0))
#     append!(e_old, check(basis_1, xds_12_1, xds_21_1))
# end



# # println("Fitting")
# fit!(basis_1, data_set_1_2)

# # println("Predicting")
# a = predict(basis_0, data_set_1_2.states[1])
# b = predict(basis_0, data_set_2_1.states[1])

# H[a2b(1), a2b(2)][b2s(6), b2s(6)]


# aᵢ, aⱼ = 1, 2

# pair_list = JuLIP.neighbourlist(atoms, 20.0)
# rr0 = project_min(atoms, atoms.X[aⱼ] - atoms.X[aᵢ])



# idxs, vecs = NeighbourLists.neigs(pair_list, i)

# env_states_o = BondState{typeof(rr0)}[BondState(v-offset, rr0, :env) for v in vecs[1:end .!= j_idx]]
# env_states_n = BondState{typeof(rr0)}[BondState(v-offset, rr0, :env) for v in vecs[1:end .!= j_idx]]
