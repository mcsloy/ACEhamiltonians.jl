# The contents of this module will be moved to a more appropriate place at a later data.
module Common
using JuLIP: Atoms
export BasisDef, parse_key, with_cache, number_of_orbitals


"""
    BasisDef(atomic_number => [ℓ₁, ..., ℓᵢ], ...)

Provides information about the basis set by specifying the azimuthal quantum numbers (ℓ)
of each shell on each species. Dictionary is keyed by atomic numbers & valued by vectors
of ℓs i.e. `Dict{atomic_number, [ℓ₁, ..., ℓᵢ]}`. 

A minimal basis set for hydrocarbon systems would be `BasisDef(1=>[0], 6=>[0, 0, 1])`.
This declares hydrogen atoms as having only a single s-shell and carbon atoms as having
two s-shells and one p-shell.
"""
BasisDef = Dict{I, Vector{I}} where I<:Integer

# Converts strings into tuples of integers or integers as appropriate. This function
# should be refactored and moved to a more appropriate location. It is mostly a hack
# at the moment.
function parse_key(key)
    if key isa Integer || key isa Tuple
        return key
    elseif '(' in key
        return Tuple(parse.(Int, split(strip(key, ['(', ')', ' ']), ", ")))
    else
        return parse(Int, key)
    end
end


"""Nᵒ of orbitals present in system `atoms` based on a given basis definition."""
function number_of_orbitals(atoms::Atoms, basis_definition::BasisDef)
    # Work out the number of orbitals on each species
    n_orbs = Dict(k=>sum(v * 2 .+ 1) for (k, v) in basis_definition)
    # Use this to get the sum of the number of orbitals on each atom
    return sum(getindex.(Ref(n_orbs), getfield.(atoms.Z, :z)))
end

"""Nᵒ of orbitals present on a specific element, `z`, based on a given basis definition."""
number_of_orbitals(z::I, basis_definition::BasisDef) where I<:Integer = sum(basis_definition[z] * 2 .+ 1)

"""
Todo:
    - Document this function correctly.

Returns a cached guarded version of a function that stores known argument-result pairs.
This reduces the overhead associated with making repeated calls to expensive functions.
It is important to note that results for identical inputs will be the same object.

# Warnings
Do no use this function, it only supports a very specific use-case.

Although similar to the Memoize packages this function's use case is quite different and
is not supported by Memoize; hence the reimplementation here. This is mostly a stop-gap
measure and will be refactored at a later data.
"""
function with_cache(func::Function)::Function
    cache = Dict()
    function cached_function(args...)
        if !haskey(cache, args)
            cache[args] = func(args...)
        end
        return cache[args]
    end
    return cached_function
end

end

