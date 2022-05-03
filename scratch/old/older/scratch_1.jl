using Serialization, Statistics
using ACEhamiltonians.Bases: Model
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

basis_pp = model.off_site_bases[(13, 13, 4, 5)]



uu = zeros(3, 3, 10)
ul = zeros(3, 3, 10)
lu = zeros(3, 3, 10)
ll = zeros(3, 3, 10)

c = 0
for i=1:5, j=1:5
    i >= j && continue
    global c += 1
    
    println(c)
    block_1_2_ref = H[a2b(i), a2b(j)][b2s(4), b2s(5)]
    block_2_1_ref = H[a2b(j), a2b(i)][b2s(5), b2s(4)]

    data_set_1_2 = collect_data(H, basis_dd, atoms, basis_def, [i, j])
    data_set_2_1 = collect_data_r(H, basis_dd, atoms, basis_def, [i, j])
    
    fit!(basis_dd, data_set_1_2)
    uu[:, :, c] = predict(basis_dd, data_set_1_2.states[1]) - block_1_2_ref
    ul[:, :, c] = predict(basis_dd, data_set_2_1.states[1]) - block_2_1_ref

    fit!(basis_dd, data_set_2_1)
    lu[:, :, c] = predict(basis_dd, data_set_1_2.states[1]) - block_1_2_ref
    ll[:, :, c] = predict(basis_dd, data_set_2_1.states[1]) - block_2_1_ref
end

uum = mean(abs.(uu))
ulm = mean(abs.(ul))
lum = mean(abs.(lu))
llm = mean(abs.(ll))





# uu = zeros(5, 5, 10)
# ul = zeros(5, 5, 10)
# lu = zeros(5, 5, 10)
# ll = zeros(5, 5, 10)

# c = 0
# for i=1:5, j=1:5
#     i >= j && continue
#     global c += 1
    
#     println(c)
#     block_1_2_ref = H[a2b(i), a2b(j)][b2s(6), b2s(6)]
#     block_2_1_ref = H[a2b(j), a2b(i)][b2s(6), b2s(6)]

#     data_set_1_2 = collect_data(H, basis_dd, atoms, basis_def, [i, j])
#     data_set_2_1 = collect_data_r(H, basis_dd, atoms, basis_def, [i, j])
    
#     fit!(basis_dd, data_set_1_2)
#     uu[:, :, c] = predict(basis_dd, data_set_1_2.states[1]) - block_1_2_ref
#     ul[:, :, c] = predict(basis_dd, data_set_2_1.states[1]) - block_2_1_ref

#     fit!(basis_dd, data_set_2_1)
#     lu[:, :, c] = predict(basis_dd, data_set_1_2.states[1]) - block_1_2_ref
#     ll[:, :, c] = predict(basis_dd, data_set_2_1.states[1]) - block_2_1_ref
# end

# uum = mean(abs.(uu))
# ulm = mean(abs.(ul))
# lum = mean(abs.(lu))
# llm = mean(abs.(ll))

function evaluate!(B, basis::Rn1pBasis, X::BondState)
    if X.be == :bond
        return evaluate!(B, basis.R, norm(X.rr))
    else
        return evaluate!(B, basis.R, norm(X.rr))
    end
end

function evaluate!(B, basis::Ylm1pBasis, X::BondState)
    if X.be  == :bond  
        return evaluate!(B, basis.SH, _rr(X, basis))
    end
end

# function evaluate!(B, basis::Rn1pBasis, X::BondState)
#     if X.be == :bond
#         return evaluate!(B, basis.R, norm(X.rr))
#     else
#         return evaluate!(B, basis.R, norm(X.rr))
#     end
# end

evaluate!(B, basis::Rn1pBasis, X::BondState) =
      evaluate!(B, basis.R, norm(_rr(X, basis)))

evaluate!(B, basis::Ylm1pBasis, X::BondState) = 
      evaluate!(B, basis.SH, _rr(X, basis))

function get_eval(basis, X)
    rnp = basis.basis.pibasis.basis1p.bases[1]
    B = zeros(8)
    val = evaluate!(B, rnp, X)
    return B
end

# Vector{ComplexF64} 81