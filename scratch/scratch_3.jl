using ACEhamiltonians
using ACEhamiltonians.Parameters: ison


path = "/home/ajmhpc/Projects/ACEtb/Data/new_test_data/build/Al3_7.h5"
basis_def = Dict(13=>[0, 0, 0, 1, 1, 2])

_shells = [0, 0, 0, 1, 1, 2]
_norbs = _shells * 2 .+ 1
_ends = cumsum(_norbs)
_starts = _ends - _norbs .+ 1
b2s(i) = _starts[i]:_ends[i]
a2b(i) = (1:14) .+ ((i - 1) * 14)

H, atoms = load_old_hamiltonian(path), load_old_atoms(path)


# ν = GlobalParams(2)
# deg = GlobalParams(2)
# e_cut_out = GlobalParams(12.0)
# e_cut_in = GlobalParams(0.01)
# b_cut = GlobalParams(12.0)

# p_on = OnSiteParaSet(ν, deg, e_cut_in, e_cut_out)

# p_off = OffSiteParaSet(ν, deg, b_cut, e_cut_in, e_cut_out)

# model = Model(basis_def, p_on, p_off);

# basis = model.off_site_bases[(6, 6, 1, 1)];

basis = OffSiteBasis(off_site_ace_basis(1, 1, 2, 4, 20.0, 12.0), (13, 13, 5,5));

ds_12 = collect_data(H, basis, atoms, basis_def, [1, 2]);

# function off_site_ace_basis(ℓ₁::I, ℓ₂::I, ν::I, deg::I, b_cut::F, e_cutₒᵤₜ::F=5., e_cutᵢₙ::F=0.5,
#     λₙ::F=.5, λₗ::F=.5) where {I<:Integer, F<:AbstractFloat}
