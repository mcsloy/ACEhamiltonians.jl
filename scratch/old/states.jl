using JuLIP, NeighbourLists, LinearAlgebra, Random, StaticArrays
using JuLIP: Atoms
using JuLIP.Utils: project_min
using ACE: CylindricalBondEnvelope
using ACEhamiltonians.DataManipulation: BondState
import ACEhamiltonians.DataManipulation: get_states
import ACE: cat2cyl





# path = "/home/ajmhpc/Projects/ACEtb/Data/test_data/FCC-MD-500K/SK-supercell-001.h5"
# atoms = load_old_atoms(path)
Random.seed!(1)
positions = ((rand(20, 3) .- 0.5) .* 20) .+ 50
positions[1, :] = [47., 50., 50.]
positions[2, :] = [53., 50., 50.]
positions[3, :] = [50., 52., 50.]


atoms = Atoms(;Z=ones(Int64, length(positions)), X=positions', cell=[100, 100, 100]);

env_0 = CylindricalBondEnvelope(18.0, 20.0, 3.0, λ=0.0, floppy=false)
env_1 = CylindricalBondEnvelope(18.0, 20.0, 3.0, floppy=false)

env_0 = CylindricalBondEnvelope(18.0, 12.0, 12.0, λ=0.0, floppy=false)
env_1 = CylindricalBondEnvelope(18.0, 12.0, 12.0, floppy=false)

states_0 = get_states(1, 2, atoms, env_0)
states_1 = get_states(1, 2, atoms, env_1)
states_2 = get_states_2(1, 2, atoms, env_1)

bond_vector = (positions[2, :] - positions[1, :])

d = positions[3, :] - (positions[1, :] + bond_vector/2)

i, j = 1, 2

pair_list = JuLIP.neighbourlist(atoms, 20.)
rr0 = project_min(atoms, atoms.X[j] - atoms.X[i])
idxs, vecs = NeighbourLists.neigs(pair_list, i)
j_idx = findfirst(x -> norm(x - rr0) < 1E-10, vecs)
vecs = vecs[1:end .!= j_idx];
