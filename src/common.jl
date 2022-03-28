# The contents of this module will be moved to a more appropriate place at a later data.
module Common

export BasisDef, parse_key


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

end

