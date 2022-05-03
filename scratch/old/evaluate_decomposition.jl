import ACE: evaluate, evaluate!
import ACEbase: evaluate, evaluate!, acquire_B!, release_B!
using ACE: CylindricalBondEnvelope, Rn1pBasis, Ylm1pBasis, _rr, _inner_evaluate, ACEConfig
using ACEhamiltonians.Bases: Basis, off_site_ace_basis, envelope
using ACEhamiltonians.DataManipulation: collect_data, collect_data_r, BondState, get_states
using ACEhamiltonians.DataIO: load_old_hamiltonian, load_old_atoms
using ACE: add_into_A!
using ACEhamiltonians

function get_B(basis, args...)
    B = acquire_B!(basis, args...)
    new_B = copy(B)
    release_B!(basis, B)
    return new_B
end



path = "/home/ajmhpc/Projects/ACEtb/Data/new_test_data/build/Al3_7.h5"
_shells = [0, 0, 0, 1, 1, 2]
_norbs = _shells * 2 .+ 1
_ends = cumsum(_norbs)
_starts = _ends - _norbs .+ 1
b2s(i) = _starts[i]:_ends[i]
a2b(i) = (1:14) .+ ((i - 1) * 14)

H, atoms = load_old_hamiltonian(path), load_old_atoms(path)
basis_def = Dict(13=>[0, 0, 0, 1, 1, 2])

env = CylindricalBondEnvelope(15.0, 12.0, 5.0, λ=0.0, floppy=false)
basis = Basis(off_site_ace_basis(1, 1, 2, 2, env, 0.1), (13, 13, 5, 5))

ds12 = collect_data(H, basis, atoms, basis_def, [1, 2]);
ds21 = collect_data_r(H, basis, atoms, basis_def, [1, 2]);

s12 = ds12.states[1];
s21 = ds21.states[1];

s12_bond = s12[1];
s21_bond = s21[1];

s12_env = s12[2];
s21_env = s21[2];

opb = basis.basis.pibasis.basis1p;
radial_basis = radial(basis);
angular_basis = angular(basis);
categorical_basis = categorical(basis);
envelope_basis = env;


cfg_12 = ACEConfig(s12);
cfg_21 = ACEConfig(s21);

A_12 = acquire_B!(opb, cfg_12); A_12 .= 0;
A_21 = acquire_B!(opb, cfg_12); A_21 .= 0;


# add_into_A!(A_12, opb, s12_env);
# add_into_A!(A_21, opb, s21_env);

# s12_bond_b = BondState(s12_bond.rr * 1.0, s12_bond.rr0, :bond);
# s21_bond_b = BondState(s21_bond.rr * -1.0, s21_bond.rr0, :bond);

A_12 = acquire_B!(opb, cfg_12); A_12 .= 0;
A_21 = acquire_B!(opb, cfg_12); A_21 .= 0;



# add_into_A!(A_12, opb, s12_bond);
# add_into_A!(A_21, opb, s21_bond);


# max(maximum(maximum(abs.(imag(A_12)))), maximum(maximum(abs.(imag(A_21)))))

# A = [real(A_12);; real(A_21)];


# size(acquire_B!(Basis(off_site_ace_basis(1, 1, 2, 2, env, 0.1), (13, 13, 5, 5)).basis.pibasis.basis1p), 1)

# ν has no effect on the size of the basis1p basis' B matrix
# deg has an effect

# ν = 2
# deg = 1:9
# 8, 26, 58, 108, 180, 278, 406, 568
# 

# 1, 4, 9, 

#maximum(reinterpret(reshape, Int64, Basis(off_site_ace_basis(2, 2, 2, 4, env, 0.1), (13, 13, 5, 5)).basis.pibasis.basis1p.indices), dims=2)[2]
