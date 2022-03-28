module DataIO
using HDF5
using HDF5: Group, h5open
using JuLIP: Atoms
using ACEhamiltonians


export load_overlap, load_hamiltonian, gamma_only, load_cell_translations, load_k_points_and_weights, load_basis_set_definition, load_atoms, load_old_hamiltonian, load_old_atoms


# Flips and array from row to column major and vice-versa 
rcf(array) = permutedims(array, reverse(1:ndims(array)))

function load_data(src::Group, target)
    data = read(src["Data/$target"])
    if data isa AbstractArray
        data = rcf(data)
    end
    return data
end

"""

Load structural information from HDF5 group into a `JuLIP.Atoms` instance.

# Todo
- Add pbc attribute to the database so that the periodic boundary conditions are known.
- Make use of unit information.
"""
function load_atoms(src::Group)
    
    geom = src["Structure"]
    z, x = read(geom["atomic_numbers"]), read(geom["positions"])

    if haskey(geom, "lattice")
        return Atoms(; Z=z, X=x, cell=rcf(read(geom["lattice"])), pbc=true)
    else
        return Atoms(; Z=z, X=x)
    end
end

"""
Loads the system's basis definition.

This will return a `BasisDef` object that lists the azimuthal quantum number of each
basis group present on each atom. 
"""
function load_basis_set_definition(src::Group)
    src = src["Info/Basis"]
    return BasisDef{Int}(parse(Int, k) => read(v)[2, :] for (k, v) in zip(keys(src), src))
end

function load_k_points_and_weights(src::Group)
    k_points_and_weights = rcf(read(src["Info/k-points"]))
    return k_points_and_weights[:, 1:3], k_points_and_weights[:, 4]
end

load_cell_translations(src::Group) = rcf(read(src["Info/Translations"]))



"""

Yields true if results are from a gamma-point only calculation.

Gamma-point only calculations will have standard two dimensional Hamiltonian and overlap
matrices. Whereas calculations with more than one k-point will have three dimensional
matrices (one slice for each k-point); they will also have a list of k-points/weights 
and cell translation vectors.
"""
gamma_only(src::Group) = !haskey(src, "Info/Translations")

load_hamiltonian(src::Group) = load_data(src, "H")
load_overlap(src::Group) = load_data(src, "S")

# Temporary working function to enable reading of deprecated databases
function load_old_hamiltonian(path::String)
    return h5open(path) do database
        read(database, "aitb/H")[:, :]
    end
end



function load_old_atoms(path::String; groupname=nothing)
    h5open(path, "r") do fd
        groupname === nothing && (groupname = HDF5.name(first(fd)))
        positions = HDF5.read(fd, string(groupname,"/positions"))
        unitcell = HDF5.read(fd, string(groupname,"/unitcell"))
        species = HDF5.read(fd, string(groupname,"/species"))
        atoms = Atoms(; X = positions, Z = species,
                        cell = unitcell,
                        pbc = [true, true, true])
        return atoms
    end
 end

end