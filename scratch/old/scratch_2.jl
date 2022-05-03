using Serialization, Statistics
using ACEhamiltonians.Bases: Model
using ACE: ACEConfig, evaluate, scaling, AbstractState
using ACEhamiltonians.Fitting2: fit!, predict!, predict
using ACEhamiltonians.DataManipulation: collect_data, collect_data_r, get_states
using ACEhamiltonians.DataIO: load_old_hamiltonian, load_old_atoms

path = "/home/ajmhpc/Projects/ACEtb/Data/test_data/FCC-MD-500K/SK-supercell-001.h5"

_shells = [0, 0, 0, 1, 1, 2]
_norbs = _shells * 2 .+ 1
_ends = cumsum(_norbs)
_starts = _ends - _norbs .+ 1

a2b(i) = (1:14) .+ ((i - 1) * 14)
b2s(i) = _starts[i]:_ends[i]

H, atoms = load_old_hamiltonian(path), load_old_atoms(path)

# block_1_2_ref = H[a2b(1), a2b(2)][b2s(6), b2s(6)]
# block_2_1_ref = H[a2b(2), a2b(1)][b2s(6), b2s(6)]

model = deserialize("Al_model_fitted.bin");
basis_def = model.basis_definition

basis_dd = model.off_site_bases[(13, 13, 6, 6)]


data_set = collect_data(H, basis_dd, atoms, basis_def, [1, 2])
states = ACEConfig(data_set.states[1])
#predict(basis_dd, data_set_1_2.states[1])


# A = evaluate(basis.basis, ACEConfig(state))
# B = evaluateval_real(A)
# values .= (basis.coefficients' * B) + basis.mean

# @generated function add_into_A!(A, basis::Product1pBasis{NB}, X) where {NB}
#     quote
#        @nexprs $NB i -> begin 
#           bas_i = basis.bases[i]
#           B_i = zeros(acquire_B!(bas_i, X))
#           evaluate!(B_i, bas_i, X)
#        end 
#        for (iA, ϕ) in enumerate(basis.indices)
#           t = one(eltype(A))
#           @nexprs $NB i -> (t *= B_i[ϕ[i]])
#        end
#        @nexprs $NB i -> release_B!(basis.bases[i], B_i)
#        return nothing
#     end
#  end

#  V1 = acquire_B!(basis_dd.basis.pibasis.basis1p, states);
#  V1 .= 0.0
#  V2 = acquire_B!(basis_dd.basis.pibasis.basis1p, states_2);
#  V2 .= 0.0

#  add_into_A!(V1, opb, collect(states)[1])
#  add_into_A!(V2, opb, collect(states_2)[1])
