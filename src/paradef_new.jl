module Parameters


using Base

export Params, GlobalParams, AtomicParams, AzimuthalParams, ShellParams, ParaSet, OnSiteParaSet, OffSiteParaSet, gather, ison

#########
# Label #
#########
# This structure is allows for order agnostic interactions representations; e.g. the
# interactions (z₁, z₂) and (z₁, z₂) are not only equivalent but are fundamentally the
# same interaction. For an atomic number pair (z₁, z₂) or a shell number pair (s₁, s₂)
# the necessary representation could be achieved using a Set. However, this starts to
# fail when describing interactions by both atomic and shell number (z₁, z₂, s₁, s₂).
# Furthermore, it is useful to be able indicate that some interactions are sub-types
# of others; i.e. all the following interactions (z₁, z₂, 1, 1), (z₁, z₂, 1, 2) and
# (z₁, z₂, 2, 2) are sub-interactions of the (z₁, z₂) interaction type. This property
# is useful when dealing with groups of interactions specifying parameters.
#
# In general it is not intended for the user to interact with `Label` entities directly.
# Such structures are primarily used in the background. IT should be noted that these
# structures are designed for ease of use rather than performance; however this should not
# be an issue given that this is never used in performance critical parts of the code.  

struct Label{N, I}
    id::NTuple{N, I}
    
    # Label(i, j, k...)
    Label(id::Vararg{I, N}) where {I<:Integer, N} = new{N, I}(_process_tuple(id))
    Label{N, I}(id::Vararg{I, N}) where {I<:Integer, N} = new{N, I}(_process_tuple(id))

    # Label((i, j, k, ...))
    Label(id::NTuple{N, I}) where {I<:Integer, N} = new{N, I}(_process_tuple(id))
    Label{N, I}(id::NTuple{N, I}) where {I<:Integer, N} = new{N, I}(_process_tuple(id))

    # Label(), Label((,)) () Special cases for an empty Label; used for global interactions
    Label(id::Tuple{}) = new{0, Int}(id)
    Label() = new{0, Int}(NTuple{0, Int}())
    Label{N, I}(id::Tuple{}) where {N, I<:Integer} = new{N, I}(id)
    Label{N, I}() where {N, I<:Integer} = new{N, I}(NTuple{N, I})
    
    # Label(Label)
    ILabel(id::T) where T<:Label = id
end

# Equality operators
Base.:(==)(x::T₁, y::T₂) where {T₁<:Label, T₂<:Label} = x.id == y.id
Base.:(==)(x::T₁, y::T₂) where {T₁<:Label, T₂<:NTuple} = x == Label(y)
Base.:(==)(x::T₁, y::T₂) where {T₁<:Label, T₂<:Integer} = x == Label(y)
Base.:(==)(x::T₁, y::T₂) where {T₁<:NTuple, T₂<:Label} = Label(x) == y
Base.:(==)(x::T₁, y::T₂) where {T₁<:Integer, T₂<:Label} = Label(x) == y

# Subset equality operators. These are used to check if one interaction label represents
# a sub-interaction type of another. For example (6, 7, 1, 1), (1, 6, 1, 2) & (1, 6, 2, 2)
# are all sub-sets of the (6, 7) interaction. Note that all interactions are subtypes of
# Label((,)) (empty label is taken to mean "all interactions").
function Base.:⊆(x::Label{N₁, I}, y::Label{N₂, I}) where {N₁, N₂, I}

    # If labels have the same length then check if they are equivalent
    if N₁≡N₂; return x==y

    # If N₁<N₂ then x cannot be equal to or a subset of y
    elseif N₁<N₂; false

    # All interactions are a subsets of the global label Label((,)); i.e. N₂≡0
    elseif N₂≡0; true

    # One and Two atom labels cannot be equal to or be subsets of one another. 
    elseif isodd(N₁)⊻isodd(N₂); false

    # Check if the x is a subset of y (one atom) 
    elseif isodd(N₁); x[1] == y[1]

    # Check if the x is a subset of y (two atom) 
    elseif iseven(N₁); x[1:2] == y[1:2]
    
    else; error("Unknown subset ⊆(Label,Label) call")
    end
end

Base.:⊆(x::T₁, y::T₂) where {T₁<:Label, T₂} = x ⊆ Label(y)
Base.:⊆(x::T₁, y::T₂) where {T₁, T₂<:Label} = Label(x) ⊆ y
Base.:⊇(x::T₁, y::T₂) where {T₁<:Label, T₂} = y ⊆ x
Base.:⊇(x::T₁, y::T₂) where {T₁, T₂<:Label} = y ⊆ x

# General utilities to aid with manipulation 
Base.show(io::IO, x::Label) = print(io, string(x.id))
Base.show(io::IO, ::Label{0, I}) where I = print(io, "Any")
Base.length(::Label{N, I}) where {N, I} = N
Base.length(::Type{Label{N, I}}) where {N, I} = N
Base.getindex(x::Label, idx) = x.id[idx]
Base.lastindex(x::Label) = lastindex(x.id)

# Conversion methods for Labels
Base.convert(t::Type{Label{N, I}}, x::NTuple{N, I}) where {N, I<:Integer} = t(x)
Base.convert(t::Type{Label{1, I}}, x::I) where I<:Integer = t(x)

##############
# Parameters #
##############
# Todo:
#   - Need to abstract AtomicParams, AzimuthalParams & ShellParams constructors
#     to a macro as they are all very similar.
#   - Need to enforce limits on key values, shells must be larger than zero and
#     azimuthal numbers must be non-negative.
#   - All Params should be combinable, with compound classes generated when combining
#     different Params types. Compound types should always check the more refined
#     struct first (i.e. ShellParams<AzimuthalParams<AtomicParams).
#   - should be specifiable with pairs AtomicParams(1=>2).


abstract type Params{K, V} end

struct GlobalParams{K, V} <: Params{K, V}
    vals::Dict{K, V}
    GlobalParams(v::V) where V = new{Label{0, Int}, V}(Dict(Label()=>v))
end


struct AtomicParams{K, V} <: Params{K, V}
    vals::Dict{K, V}

    AtomicParams(v::Dict{K, V}) where {K<:Label{1, I}, V} where I<:Integer = new{K, V}(v)
    AtomicParams(v::Dict{K, V}) where {K<:Label{2, I}, V} where I<:Integer = new{K, V}(v)
    AtomicParams(v::Dict{NTuple{N, I}, V}) where {N, I<:Integer, V} = AtomicParams(
        _guarded_convert(Dict{Label{N, I}, V}, v))
    AtomicParams(v::Dict{I, V}) where {I, V} = AtomicParams(
        _guarded_convert(Dict{Label{1, I}, V}, v))

end

# Need to enforce that azimuthal numbers are 0 or grater
struct AzimuthalParams{K, V} <: Params{K, V}
    vals::Dict{K, V}

    Azimuthal(v::Dict{K, V}) where {K<:Label{3, I}, V} where I = new{K, V}(v)
    Azimuthal(v::Dict{K, V}) where {K<:Label{4, I}, V} where I = new{K, V}(v)
    Azimuthal(v::Dict{NTuple{N, I}, V}) where {N, I, V} = Azimuthal(
        _guarded_convert(Dict{Label{N, I}, V}, v))
end

# Need to enforce that shell numbers are positive but none zero
struct ShellParams{K, V} <: Params{K, V}
    vals::Dict{K, V}

    ShellParams(v::Dict{K, V}) where {K<:Label{3, I}, V} where I = new{K, V}(v)
    ShellParams(v::Dict{K, V}) where {K<:Label{4, I}, V} where I = new{K, V}(v)
    ShellParams(v::Dict{NTuple{N, I}, V}) where {N, I, V} = ShellParams(
        _guarded_convert(Dict{Label{N, I}, V}, v))
end

# General utilities
Base.valtype(::Params{K, V}) where {K, V} = V
Base.keytype(::Params{K, V}) where {K, V} = K
Base.valtype(::Type{Params{K, V}}) where {K, V} = V
Base.keytype(::Type{Params{K, V}}) where {K, V} = K
Base.keys(x::Params) = keys(x.vals)
Base.values(x::Params) = values(x.vals)

Base.length(x::T) where T<:Params = length(x.vals)


# IO
Base.show(io::IO, x::GlobalParams) = print(io, "GlobalParams{$(valtype(x))}($(x.vals))")

"""Full, multi-line string representation of a `Param` type objected"""
function _multi_line(x::T) where T<:Params
    i = length(keytype(x.vals).types[1].types) ≡ 1 ? 1 : Base.:(:)
    v_string = join(["$(k[i]) => $v" for (k, v) in x.vals], "\n  ")
    return "$(nameof(T)){$(valtype(x))} with $(length(x.vals)) entries:\n  $(v_string)"
end


function Base.show(io::O, x::T) where {T<:Params, O<:IO}
    # If printing an isolated Params instance, just use the standard multi-line format
    if !haskey(io.dict, :SHOWN_SET)
        print(io, _multi_line(x))
    # If the Params is being printed as part of a group then a more compact
    # representation is needed.
    else
        # Create a slicer remove braces from tuples of length 1 if needed
        s = length(keytype(v)) == 1 ? 1 : Base.:(:)
        # Sort the keys to ensure consistency
        keys_s = sort([j.id for j in keys(x.vals)])  
        # Only show first and last keys (or just the first if there is only one)
        targets = length(x) != 1 ? [[1, lastindex(keys_s)]] : [1]
        # Build the key list and print the message out
        k_string = join([k[s] for k in keys_s[targets...]], " … ")
        print(io, "$(nameof(T))($(k_string))")
    end
end

# Special show case: Needed as Base.TTY has no information dictionary 
Base.show(io::Base.TTY, x::T) where T<:Params = print(io, _multi_line(x))

# Indexing operations
function Base.getindex(x::T, key) where T<:GlobalParams
    # Global parameters will always return the same value irrespective of what
    # key is given. The only condition is that the key is of a valid form. 
    Label(key) # ← only invoked to ensure that key is a valid form 
    return first(values(x))
end

function Base.getindex(x::T, key) where T<:Params
    # This will not only match the specified key but also any superset it is a part of;
    # i.e. the key (z₁, z₂, s₁, s₂) will match (z₁, z₂).
    super_key = filter(k->(key ⊆ k), keys(x))
    if length(super_key) ≡ 0
        throw(KeyError(key))
    else
        return x.vals[first(super_key)]
    end
end

###########
# ParaSet #
###########
# Todo:
#   - Abstract error checking.
#   - Ensure that OnSiteParaSet is provided with correctly keyed Params instances.
#   - Ensure that outer cutoff is greater than the inner cutoff

abstract type ParaSet end


struct OnSiteParaSet <: ParaSet
    ν
    deg
    e_cut_out
    e_cut_in

    function OnSiteParaSet(ν::T₁, deg::T₂, e_cut_out::T₃, e_cut_in::T₄
        ) where {T₁<:Params, T₂<:Params, T₃<:Params, T₄<:Params}
        @assert valtype(ν)<:Integer "ν must be an integer"
        @assert valtype(deg)<:Integer "deg must be an integer"
        @assert valtype(e_cut_out)<:AbstractFloat "e_cut_out must be a float"
        @assert valtype(e_cut_in)<:AbstractFloat "e_cut_in must be an float"
        new(ν, deg, e_cut_out, e_cut_in)
    end

end

struct OffSiteParaSet <: ParaSet
    ν
    deg
    b_cut
    e_cut_out
    e_cut_in
    
    function OffSiteParaSet(ν::T₁, deg::T₂, b_cut::T₃, e_cut_out::T₄, e_cut_in::T₅
        ) where {T₁<:Params, T₂<:Params, T₃<:Params, T₄<:Params, T₅<:Params}
        @assert valtype(ν)<:Integer "ν must be an integer"
        @assert valtype(deg)<:Integer "deg must be an integer"
        @assert valtype(b_cut)<:AbstractFloat "e_cut_out must be a float"
        @assert valtype(e_cut_out)<:AbstractFloat "e_cut_in must be an float"
        @assert valtype(e_cut_in)<:AbstractFloat "e_cut_in must be an float"
        new(ν, deg, e_cut_out, e_cut_in, b_cut)
    end

end


### External helper functions
# Todo:
#   - This should be replaced with something a little more elegant.


ison(::OnSiteParaSet) = true
ison(::OffSiteParaSet) = false

function gather(para::OnSiteParaSet, id)
    # Todo: document this
    return (f -> getfield(para, f)[id]).(
        (:ν, :deg, :e_cut_out, :e_cut_in))
end

function gather(para::OffSiteParaSet, id)
    # Todo: document this
    return (f -> getfield(para, f)[id]).(
        (:ν, :deg, :b_cut, :e_cut_out, :e_cut_in))
end    


#############################
# Internal Helper Functions #
#############################

# Guards type conversion of dictionaries keyed with `Label` entities. This is done to
# ensure that a meaningful message is given to the user when a key-collision occurs.
function _guarded_convert(t::Type{Dict{Label{N, I}, V}}, x::Dict{NTuple{N, I}, V}) where {N, I<:Integer, V}
    try
        return convert(t, x)
    catch e
        if e.msg == "key collision during dictionary conversion" 
            r_keys = _redundant_keys([k for k in keys(x)])
            error("Redundant keys found:\n$(join(["  - $(join(i, ", "))" for i in r_keys], "\n"))")
        else
            rethrow(e)
        end
    end
end

# Collisions cannot occur when input dictionary is keyed by integers not tuples
_guarded_convert(t::Type{Dict{Label{1, I}, V}}, x::Dict{I, V}) where {N, I<:Integer, V} = convert(t, x)


"""
Sort `Label` tuples so that the lowest atomic-number/shell-number comes first for the
two/one atom interaction labels. If more than four integers are specified then an error
is raised.
"""
function _process_tuple(x::NTuple{N, I}) where {N, I<:Integer}
    if N <= 1; x
    elseif N ≡ 2; x[1] ≤ x[2] ? x : reverse(x)
    elseif N ≡ 3; x[2] ≤ x[3] ? x : x[[1, 3, 2]]
    elseif N ≡ 4
        if x[1] > x[2] || ((x[1] ≡ x[2]) && (x[3] > x[4])); x[[2, 1, 4, 3]]
        else; x
        end
    else
        error(
            "Label may contain no more than four integers, valid formats are:\n"*
            "  ()\n  (z₁,)\n  (z₁, s₁, s₂)\n  (z₁, z₂)\n  (z₁, z₂, s₁, s₂)")
    end
end

function _redundant_keys(keys_in::Vector{NTuple{N, I}}) where {I<:Integer, N}
    duplicates = []
    while length(keys_in) ≥ 1
        key = Label(pop!(keys_in))
        matches = [popat!(keys_in, i) for i in findall(i -> i == key, keys_in)]
        if length(matches) ≠ 0
            append!(duplicates, Ref((key, matches...)))
        end
    end
    return duplicates
end

end