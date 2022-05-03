module Interface

using Base, PyCall, JuLIP, Serialization
using ACEbase: read_dict, load_json
using ACEhamiltonians.Bases: Model
using ACEhamiltonians.Fitting2: predict

export cache_model, uncache_model, evaluate

ase = pyimport("ase")

# Functions for interconverting between `ase.Atoms` and `JuLIP.Atoms` objects. Note, these
# should be replaced with non-base functions to prevent collisions with other `PyObject`
# instances. 
Base.convert(::Type{JuLIP.Atoms}, o::PyObject) = JuLIP.Atoms(
    ;Z = o.get_atomic_numbers(), X = o.positions', cell = o.cell.array, pbc = o.pbc)

Base.convert(::PyObject, o::JuLIP.Atoms) = ase.Atoms(
    getfield.(o.Z, :z), o.X, cell=o.cell, pbc=o.pbc)


# There is a non-trivial performance bottleneck associated with process of loading and
# instantiating ACEhamiltonians models from the JSON files in which they are stored. The
# time required commonly exceeds that needed to actually evaluate the model on a target
# system. Thus, loaded models are cached in the `_LOADED_MODELS` dictionary. Models can
# be pre-loaded via the `cache_model` function and unloaded via `uncache_model`.
_LOADED_MODELS = Dict{String, Model}()

"""Add a model to the cache to speed up access."""
function cache_model(path::String)
    if !haskey(_LOADED_MODELS, path)
        if endswith(lowercase(path), "json")
            _LOADED_MODELS[path] = read_dict(load_json(path))
        elseif endswith(lowercase(path), "bin")
            _LOADED_MODELS[path] = deserialize(path)
        else
            error("Unknown file extension used; only json & bin are supported")
        end
    end
    nothing
end

"""Remove a model from the cache to free up memory."""
function uncache_model(path::String)
    if haskey(_LOADED_MODELS, path)
        delete!(_LOADED_MODELS, path)
    end
    nothing
end

"""Load a model from a JSON file or retrieve a cached version of it."""
function retrieve_model(path::String)
    if !haskey(_LOADED_MODELS, path)
        cache_model(path)
    end
    return _LOADED_MODELS[path]

end


# Evaluate the supplied model on the target system
evaluate(model_path::String, atoms::PyObject) = convert(
    PyObject,
    predict(
        # Load the model (from the cache if possible)
        retrieve_model(model_path),
        # Convert from ase.Atoms to JuLIP.Atoms
        convert(JuLIP.Atoms, atoms))
        )

end