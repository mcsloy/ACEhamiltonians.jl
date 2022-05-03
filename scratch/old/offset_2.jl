using Serialization, Statistics, LinearAlgebra, JuLIP, NeighbourLists
using JuLIP.Utils: project_min
import ACE: evaluate!
import ACEbase: evaluate!
using ACEhamiltonians.Bases: Basis, off_site_ace_basis, envelope
using ACEhamiltonians.DataIO: load_old_hamiltonian, load_old_atoms
using ACEhamiltonians.DataManipulation: collect_data, collect_data_r, get_states, DataSet, BondState
using ACE: CylindricalBondEnvelope, Rn1pBasis, Ylm1pBasis, _rr, _inner_evaluate
using ACEhamiltonians.Fitting2: fit!, predict
using ACE: cat2cyl, _eff_zcut, _eval_env_inner, _inner_evaluate

function check_block(basis, s₁, s₂, i, j)
    u = predict(basis, s₁)
    l = predict(basis, s₂)
    ur = H[a2b(i), a2b(j)][b2s(6), b2s(6)]
    lr = H[a2b(j), a2b(i)][b2s(6), b2s(6)]
    e1 = mean(abs.(u - ur))
    e2 = mean(abs.(l - lr))
    e3 = mean(abs.(u - l'))
    e4 = mean(abs.(u - l))
    return (e1, e2, e3, e4)
end


function blocks(basis, s₁, s₂)
    return predict(basis, s₁), predict(basis, s₂)
end



path = "/home/ajmhpc/Projects/ACEtb/Data/test_data/FCC-MD-500K/SK-supercell-001.h5"
_shells = [0, 0, 0, 1, 1, 2]
_norbs = _shells * 2 .+ 1
_ends = cumsum(_norbs)
_starts = _ends - _norbs .+ 1
b2s(i) = _starts[i]:_ends[i]
a2b(i) = (1:14) .+ ((i - 1) * 14)

println("Collecting data")
H, atoms = load_old_hamiltonian(path), load_old_atoms(path)
basis_def = Dict(13=>[0, 0, 0, 1, 1, 2])


if isfile("basis_dd_new.bin")
    println("Loading model")
    basis = deserialize("basis_dd_new.bin")
    env = envelope(basis)
else
    println("Building model")
    env = CylindricalBondEnvelope(18.0, 12.0, 5.0, λ=0.0, floppy=false)
    basis = Basis(off_site_ace_basis(2, 2, 2, 8, env, 0.1), (13, 13, 6, 6))
    serialize("basis_dd_new.bin", basis)
end


function state_mod_1(data_set)
    return DataSet(
        data_set.values, data_set.block_indices, data_set.cell_indices,
        [[BondState(x[1].rr ./ 2, x[1].rr0, :bond); x[2:end]] for x in data_set.states]
        )
end


function state_mod_2(data_set)
    return DataSet(
        data_set.values, data_set.block_indices, data_set.cell_indices,
        [[BondState(-x[1].rr ./ 2, x[1].rr0, :bond); x[2:end]] for x in data_set.states]
        )
end

ds_12 = collect_data(H, basis, atoms, basis_def, [1, 2]);
ds_21 = collect_data_r(H, basis, atoms, basis_def, [1, 2]);

ds_23 = collect_data(H, basis, atoms, basis_def, [2, 3]);
ds_32 = collect_data_r(H, basis, atoms, basis_def, [2, 3]);

check_block(basis, ds_12.states[1], ds_21.states[1], 1, 2)

ds_12a = state_mod_1(ds_12)
ds_21a = state_mod_1(ds_21)
ds_12b = state_mod_2(ds_12)
ds_21b = state_mod_2(ds_21)


fit!(basis, ds_12); check_block(basis, ds_12.states[1], ds_21.states[1], 1, 2)
fit!(basis, ds_12a); check_block(basis, ds_12a.states[1], ds_21a.states[1], 1, 2)
fit!(basis, ds_12b); check_block(basis, ds_12b.states[1], ds_21b.states[1], 1, 2)

