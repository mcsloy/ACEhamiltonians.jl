using Base, PyCall, JuLIP, Serialization, DelimitedFiles
using ACEbase: read_dict, load_json
using ACEhamiltonians.Bases: Model
using ACEhamiltonians.Fitting2: predict

ase = pyimport("ase")
pyimport("ase.io")

_load_geometry(path::String) = let a = ase.io.read(path)
    JuLIP.Atoms(;Z=a.get_atomic_numbers(), X=a.positions',
    cell=a.cell.array, pbc=a.pbc)
end

function _load_model(path::String)
    if endswith(lowercase(path), ".json")
        return read_dict(load_json(path))
    elseif endswith(lowercase(path), ".bin")
        return deserialize(path)
    else
        error("Unknown file extension used; only \"json\" & \"bin\" are supported")
    end
end

writedlm(ARGS[3], predict(
    _load_model(ARGS[1]),
    _load_geometry(ARGS[2])),
',');
