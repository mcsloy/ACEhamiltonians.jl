module Bases
# using HDF5
using HDF5: Group
using ACE, SparseArrays
using ACE: SymmetricBasis, SphericalMatrix, Utils.RnYlm_1pbasis, SimpleSparseBasis,
           CylindricalBondEnvelope, Categorical1pBasis, cutoff_radialbasis
using ACEbase
import ACEbase: read_dict, write_dict
using ACEbase.ObjectPools: VectorPool
using ACEhamiltonians
using ACEhamiltonians.Parameters: OnSiteParaSet, OffSiteParaSet





export Basis, IsoBasis, AnisoBasis, radial, angular, categorical, envelope, Model, on_site_ace_basis, off_site_ace_basis, filter_offsite_be, is_fitted
"""
TODO:
    - Figure out what is going on with filter_offsite_be and its arguments.
    - A warning should perhaps be given if no filter function is given when one is
      expected; such as off-site functions. If no-filter function is desired than
      a dummy filter should be required.
    - Improve typing for the Model structure.
    - Replace e_cutₒᵤₜ, e_cutᵢₙ, etc. with more "typeable" names.
"""
###################
# Basis Structure #
###################
# Todo:
#   - Document
#   - Give type information
#   - Serialization routines
#   - `AnisoBasis` instances should raise an error if it is used inappropriately.


# ╔═══════╗
# ║ Basis ║
# ╚═══════╝

abstract type Basis{T} end


"""


ACE basis for modelling symmetry invariant interactions.


# Fields
- `basis::SymmetricBasis`:
- `id::Tuple`:
- `coefficients::Vector`:
- `mean::Matrix`:

"""
struct IsoBasis{T} <: Basis{T}
    basis::SymmetricBasis
    id
    coefficients
    mean

    function IsoBasis(basis, id)
        t = ACE.valtype(basis)
        F = real(t.parameters[5])
        new{F}(basis, id, zeros(F, length(basis)), zeros(F, size(zero(t))))
    end

    function IsoBasis(basis, id, coefficients, mean)
        type = real(ACE.valtype(basis).parameters[5])
        new{type}(basis, id, coefficients, mean)
    end

end


"""

ACE basis for modelling symmetry variant interactions.


- `basis::SymmetricBasis`:
- `basis_i::SymmetricBasis`:
- `id::Tuple`:
- `coefficients::Vector`:
- `coefficients_i::Vector`:
- `mean::Matrix`:
- `mean_i::Matrix`:

"""

struct AnisoBasis{T} <: Basis{T}
    basis
    basis_i
    id
    coefficients
    coefficients_i
    mean
    mean_i

    function AnisoBasis(basis, basis_i, id)
        t₁, t₂ = ACE.valtype(basis), ACE.valtype(basis_i)
        F = real(t₁.parameters[5])
        new{F}(
            basis, basis_i,  id, zeros(F, length(basis)), zeros(F, length(basis_i)),
            zeros(F, size(zero(t₁))), zeros(F, size(zero(t₂))))
    end

    function AnisoBasis(basis, basis_i, id, coefficients, coefficients_i, mean, mean_i)
        type = real(ACE.valtype(basis).parameters[5])
        new{type}(basis, basis_i,  id, coefficients, coefficients_i, mean, mean_i)
    end
end


Basis(basis, id) = IsoBasis(basis, id)
Basis(basis, basis_i, id) = AnisoBasis(basis, basis_i, id)

# ╭───────┬───────────────────────╮
# │ Basis │ General Functionality │
# ╰───────┴───────────────────────╯ 
"""Boolean indicating whether a `Basis` instance is fitted; i.e. has non-zero coefficients"""
is_fitted(basis::IsoBasis) = !all(basis.coefficients .≈ 0.0)
is_fitted(basis::AnisoBasis) = !(
    all(basis.coefficients .≈ 0.0) && all(basis.coefficients_i .≈ 0.0))

"""Check if two `Basis` instances are equivalent"""
function Base.:(==)(x::T₁, y::T₂) where {T₁<:Basis, T₂<:Basis}
    if T₁ != T₂ || typeof(x.basis) != typeof(y.basis)
        return false
    elseif T₁<:AnisoBasis && (typeof(x.basis_i) != typeof(y.basis_i))
        return false
    else
        return all(getfield(x, i) == getfield(y, i) for i in fieldnames(T₁))
    end
end


"""Expected shape of the sub-block associated with the `Basis`; 3×3 for a pp basis etc."""
Base.size(basis::Basis) = (ACE.valtype(basis.basis).parameters[3:4]...,)

"""Expected type of resulting sub-blocks."""
Base.valtype(::Basis{T}) where T = T

"""Azimuthal quantum numbers associated with the `Basis`."""
azimuthals(basis::Basis) = (ACE.valtype(basis.basis).parameters[1:2]...,)

"""Returns a boolean indicating if the basis instance represents an on-site interaction."""
Parameters.ison(x::Basis) = length(x.id) ≡ 3


"""
    _filter_bases(basis, type)

Helper function to retrieve specific basis function information out of a `Basis` instance.
This is an internal function which is not expected to be used outside of this module. 

Arguments:
- `basis::Basis`: basis instance from which function is to be extracted.
- `type::DataType`: type of the basis functions to extract; e.g. `CylindricalBondEnvelope`.
"""
function _filter_bases(basis::Basis, T)
    functions = filter(i->i isa T, basis.basis.pibasis.basis1p.bases)
    if length(functions) == 0
        error("Could not locate basis function matching the supplied type")
    elseif length(functions) ≥ 2
        @warn "Multiple matching basis functions found, only the first will be returned"
    end
    return functions[1]
end

"""Extract and return the radial component of a `Basis` instance."""
radial(basis::Basis) = _filter_bases(basis, ACE.Rn1pBasis)

"""Extract and return the angular component of a `Basis` instance."""
angular(basis::Basis) = _filter_bases(basis, ACE.Ylm1pBasis)

"""Extract and return the categorical component of a `Basis` instance."""
categorical(basis::Basis) = _filter_bases(basis, ACE.Categorical1pBasis)

"""Extract and return the bond envelope component of a `Basis` instance."""
envelope(basis::Basis) = _filter_bases(basis, ACE.BondEnvelope)


# ╭───────┬──────────────────╮
# │ Basis │ IO Functionality │
# ╰───────┴──────────────────╯
"""
    write_dict(basis[,hash_basis])

Convert an `IsoBasis` structure instance into a representative dictionary.

# Arguments
- `basis::IsoBasis`: the `IsoBasis` instance to parsed.
- `hash_basis::Bool`: ff `true` then hash values will be stored in place of
  the `SymmetricBasis` objects.
"""
function write_dict(basis::T, hash_basis=false) where T<:IsoBasis
    return Dict(
        "__id__"=>"IsoBasis",
        "basis"=>hash_basis ? string(hash(basis.basis)) : write_dict(basis.basis),
        "id"=>basis.id,
        "coefficients"=>write_dict(basis.coefficients),
        "mean"=>write_dict(basis.mean))

end


"""
    write_dict(basis[,hash_basis])

Convert an `AnisoBasis` structure instance into a representative dictionary.

# Arguments
- `basis::AnisoBasis`: the `AnisoBasis` instance to parsed.
- `hash_basis::Bool`: ff `true` then hash values will be stored in place of
  the `SymmetricBasis` objects.
"""
function write_dict(basis::T, hash_basis::Bool=false) where T<:AnisoBasis
    return Dict(
        "__id__"=>"AnisoBasis",
        "basis"=>hash_basis ? string(hash(basis.basis)) : write_dict(basis.basis),
        "basis_i"=>hash_basis ? string(hash(basis.basis_i)) : write_dict(basis.basis_i),
        "id"=>basis.id,
        "coefficients"=>write_dict(basis.coefficients),
        "coefficients_i"=>write_dict(basis.coefficients_i),
        "mean"=>write_dict(basis.mean),
        "mean_i"=>write_dict(basis.mean_i))
end

"""Instantiate an `IsoBasis` instance from a representative dictionary."""
function ACEbase.read_dict(::Val{:IsoBasis}, dict::Dict)
    return IsoBasis(
        read_dict(dict["basis"]),
        Tuple(dict["id"]),
        read_dict(dict["coefficients"]),
        read_dict(dict["mean"]))
end

"""Instantiate an `AnisoBasis` instance from a representative dictionary."""
function ACEbase.read_dict(::Val{:AnisoBasis}, dict::Dict)
    return AnisoBasis(
        read_dict(dict["basis"]),
        read_dict(dict["basis_i"]),
        Tuple(dict["id"]),
        read_dict(dict["coefficients"]),
        read_dict(dict["coefficients_i"]),
        read_dict(dict["mean"]),
        read_dict(dict["mean_i"]))
end


function Base.show(io::IO, basis::T) where T<:Basis
    print(io, "$(nameof(T))(id: $(basis.id), fitted: $(is_fitted(basis))")
end



# ╔═══════╗
# ║ Model ║
# ╚═══════╝

struct Model
    on_site_bases
    off_site_bases

    on_site_parameters::OnSiteParaSet
    off_site_parameters::OffSiteParaSet
    basis_definition

    function Model(on_site_bases, off_site_bases,
        on_site_parameters::OnSiteParaSet, off_site_parameters::OffSiteParaSet, basis_definition)
        new(on_site_bases, off_site_bases, on_site_parameters, off_site_parameters, basis_definition)
    end
    
    function Model(basis_definition::BasisDef, on_site_parameters::OnSiteParaSet,
                   off_site_parameters::OffSiteParaSet)
        # Developers Notes
        # This makes the assumption that all z₁-z₂-ℓ₁-ℓ₂ interactions are represented
        # by the same model.
        # Discuss use of the on/off_site_cache entities

        on_sites = Dict{NTuple{3, keytype(basis_definition)}, Basis}()
        off_sites = Dict{NTuple{4, keytype(basis_definition)}, Basis}()
        
        # Caching the basis functions of the functions is faster and allows ust to reuse
        # the same basis function for similar interactions.
        ace_basis_on = with_cache(on_site_ace_basis)
        ace_basis_off = with_cache(off_site_ace_basis)

        # Sorting the basis definition makes avoiding interaction doubling easier.
        # That is to say, we don't create models for both H-C and C-H interactions
        # as they represent the same thing.
        basis_definition_sorted = sort(collect(basis_definition), by=first)
        
        # Loop over all unique species pairs then over all combinations of their shells. 
        for (zₙ, (zᵢ, shellsᵢ)) in enumerate(basis_definition_sorted)
            for (zⱼ, shellsⱼ) in basis_definition_sorted[zₙ:end]
                homo_atomic = zᵢ == zⱼ
                for (n₁, ℓ₁) in enumerate(shellsᵢ), (n₂, ℓ₂) in enumerate(shellsⱼ)
                    homo_orbital = n₁ == n₂

                    # Skip symmetrically equivalent interactions. 
                    homo_atomic && n₁ > n₂ && continue
                    
                    if homo_atomic
                        id = (zᵢ, n₁, n₂)
                        ace_basis = ace_basis_on( # On-site bases
                            ℓ₁, ℓ₂, on_site_parameters[id]...)

                        on_sites[(zᵢ, n₁, n₂)] = Basis(ace_basis, id)
                    end

                    id = (zᵢ, zⱼ, n₁, n₂)
                    ace_basis = ace_basis_off( # Off-site bases
                        ℓ₁, ℓ₂, off_site_parameters[id]...)
                    
                    # Unless this is a homo-atomic homo-orbital interaction a double basis
                    # is needed.
                    if homo_atomic && homo_orbital
                        off_sites[(zᵢ, zⱼ, n₁, n₂)] = Basis(ace_basis, id)
                    else
                        ace_basis_i = ace_basis_off(
                            ℓ₂, ℓ₁, off_site_parameters[(zⱼ, zᵢ, n₂, n₁)]...)
                        off_sites[(zᵢ, zⱼ, n₁, n₂)] = Basis(ace_basis, ace_basis_i, id)
                    end
                end
            end
        end

    new(on_sites, off_sites, on_site_parameters, off_site_parameters, basis_definition)
    end

end

# Associated methods

Base.:(==)(x::Model, y::Model) = (
    x.on_site_bases == y.on_site_bases && x.off_site_bases == y.off_site_bases
    && x.on_site_parameters == y.on_site_parameters && x.off_site_parameters == y.off_site_parameters)


# ╭───────┬──────────────────╮
# │ Model │ IO Functionality │
# ╰───────┴──────────────────╯

function ACEbase.write_dict(m::Model)
    # ACE bases are stored as hash values which are checked against the "bases_hashes"
    # dictionary during reading. This avoids saving multiple copies of the same object;
    # which is common as `Basis` objects tend to share basis functions.


    bases_hashes = Dict{String, Any}()

    function add_basis(basis)
        # Store the hash/basis pair in the bases_hashes dictionary. As the `write_dict`
        # method can be quite costly to evaluate it is best to only call it when strictly
        # necessary; hence this function exists.
        basis_hash = string(hash(basis))
        if !haskey(bases_hashes, basis_hash)
            bases_hashes[basis_hash] = write_dict(basis)
        end
    end

    for basis in union(values(m.on_site_bases), values(m.off_site_bases))        
        add_basis(basis.basis)
        if basis isa AnisoBasis
            add_basis(basis.basis_i)
        end
    end

    dict =  Dict(
        "__id__"=>"HModel",
        "on_site_bases"=>Dict(k=>write_dict(v, true) for (k, v) in m.on_site_bases),
        "off_site_bases"=>Dict(k=>write_dict(v, true) for (k, v) in m.off_site_bases),
        "on_site_parameters"=>write_dict(m.on_site_parameters),
        "off_site_parameters"=>write_dict(m.off_site_parameters),
        "basis_definition"=>Dict(k=>write_dict(v) for (k, v) in m.basis_definition),
        "bases_hashes"=>bases_hashes)
    
    return dict
end


function ACEbase.read_dict(::Val{:HModel}, dict::Dict)::Model

    function set_bases(target, basis_functions)
        for v in values(target)
            v["basis"] = basis_functions[v["basis"]]
            if v["__id__"] == "AnisoBasis"
                v["basis_i"] = basis_functions[v["basis_i"]]
            end
        end
    end

    # Replace basis object hashs with the appropriate object. 
    
    set_bases(dict["on_site_bases"], dict["bases_hashes"])
    set_bases(dict["off_site_bases"], dict["bases_hashes"])

    ensure_int(v) = v isa String ? parse(Int, v) : v
    
    return Model(
        Dict(parse_key(k)=>read_dict(v) for (k, v) in dict["on_site_bases"]),
        Dict(parse_key(k)=>read_dict(v) for (k, v) in dict["off_site_bases"]),
        read_dict(dict["on_site_parameters"]),
        read_dict(dict["off_site_parameters"]),
        Dict(ensure_int(k)=>read_dict(v) for (k, v) in dict["basis_definition"]))
end


# Todo: this is mostly to stop terminal spam and should be updated
# with more meaningful information later on.
function Base.show(io::IO, model::Model)

    # Work out if the on/off site bases are fully, partially or un-fitted.
    f = b -> if all(b) "no" elseif all(!, b) "yes" else "partially" end
    on = f([!is_fitted(i) for i in values(model.on_site_bases)])
    off = f([!is_fitted(i) for i in values(model.off_site_bases)])
    
    # Identify the species present
    species = join(sort(unique(getindex.(collect(keys(model.on_site_bases)), 1))), ", ", " & ")

    print(io, "Model(fitted=(on: $on, off: $off), species: ($species))")
end


# ╔════════════════════════╗
# ║ ACE Basis Constructors ║
# ╚════════════════════════╝

@doc raw"""

    on_site_ace_basis(ℓ₁, ℓ₂, ν, deg, e_cutₒᵤₜ[, e_cutᵢₙ])

Initialise a simple on-site `SymmetricBasis` instance with sensible default parameters.

The on-site `SymmetricBasis` entities are produced by applying a `SimpleSparseBasis`
selector to a `Rn1pBasis` instance. The latter of which is initialised via the `Rn_basis`
method, using all the defaults associated therein except `e_cutₒᵤₜ` and `e_cutᵢₙ` which are
provided by this function. This facilitates quick construction of simple on-site `Bases`
instances; if more fine-grain control over over the initialisation process is required
then bases must be instantiated manually. 

# Arguments
- `(ℓ₁,ℓ₂)::Integer`: azimuthal numbers of the basis function.
- `ν::Integer`: maximum correlation order.
- `deg::Integer`: maximum polynomial degree.
- `e_cutₒᵤₜ::AbstractFloat`: only atoms within the specified cutoff radius will contribute
   to the local environment.
- `e_cutᵢₙ::AbstractFloat`: inner cutoff radius, defaults to 2.5.

# Returns
- `basis::SymmetricBasis`: ACE basis entity for modelling the specified interaction. 

"""
function on_site_ace_basis(ℓ₁::I, ℓ₂::I, ν::I, deg::I, e_cutₒᵤₜ::F, e_cutᵢₙ::F=2.5
    ) where {I<:Integer, F<:AbstractFloat}
    # Build i) a matrix indicating the desired sub-block shape, ii) the one
    # particle Rₙ·Yₗᵐ basis describing the environment, & iii) the basis selector.
    # Then instantiate the SymmetricBasis required by the Basis structure.
    return SymmetricBasis(
        SphericalMatrix(ℓ₁, ℓ₂; T=ComplexF64),
        RnYlm_1pbasis(maxdeg=deg, r0=e_cutᵢₙ, rcut=e_cutₒᵤₜ),
        SimpleSparseBasis(ν, deg))
end

@doc raw"""

    off_site_ace_basis(ℓ₁, ℓ₂, ν, deg, b_cut[,e_cutₒᵤₜ, e_cutᵢₙ, λₙ, λₗ])


Initialise a simple off-site `SymmetricBasis` instance with sensible default parameters.

Operates similarly to [`on_site_ace_basis`](@ref) but applies a `CylindricalBondEnvelope` to
the `Rn1pBasis` basis instance. The length and radius of the cylinder are defined as
maths: ``b_{cut}+2e_{cut\_out}`` and maths: ``e_{cut\_out}`` respectively; all other
parameters resolve to their defaults as defined by their constructors. Again, instances
must be manually instantiated if more fine-grained control is desired.

# Arguments
- `(ℓ₁,ℓ₂)::Integer`: azimuthal numbers of the basis function.
- `ν::Integer`: maximum correlation order.
- `deg::Integer`: maximum polynomial degree.
- `b_cut::AbstractFloat`: cutoff distance for bonded interactions.
- `e_cutₒᵤₜ::AbstractFloat`: radius and axial-padding of the cylindrical bond envelope that
   is used to determine which atoms impact to the bond's environment.
- `e_cutᵢₙ::AbstractFloat`: inner cutoff radius, defaults to 2.5.
- `λₙ::AbstractFloat`: ???
- `λₗ::AbstractFloat`: ???

# Returns
- `basis::SymmetricBasis`: ACE basis entity for modelling the specified interaction. 

"""
function off_site_ace_basis(ℓ₁::I, ℓ₂::I, ν::I, deg::I, b_cut::F, e_cutₒᵤₜ::F=5., e_cutᵢₙ::F=0.05,
    λₙ::F=.5, λₗ::F=.5) where {I<:Integer, F<:AbstractFloat}

    # Bond envelope which controls which atoms are seen by the bond.
    env = CylindricalBondEnvelope(b_cut, e_cutₒᵤₜ, e_cutₒᵤₜ, floppy=false, λ=0.0)

    # Categorical1pBasis is applied to the basis to allow atoms which are part of the
    # bond to be treated differently to those that are just part of the environment.
    discriminator = Categorical1pBasis([:bond, :env]; varsym=:be, idxsym=:be)

    # The basis upon which the above entities act.
    RnYlm = RnYlm_1pbasis(maxdeg=deg, r0=e_cutᵢₙ, rcut=cutoff_radialbasis(env))
    
    return SymmetricBasis(
        SphericalMatrix(ℓ₁, ℓ₂; T=ComplexF64),
        RnYlm * env * discriminator,
        SimpleSparseBasis(ν + 1, deg),
        filterfun=bb -> filter_offsite_be(bb, deg, λₙ, λₗ))
end

 
"""
    filter_offsite_be(bb, deg[, λₙ=0.5, λₗ=0.5])

Some mysterious filtering function.

This filter function should be passed, via the keyword `filterfun`, to `SymmetricBasis`
when instantiating.


# Arguments
- `bb:?????`: Unknown, this is supplied by ase when used?
- `deg::Integer`: maximum polynomial degree.
- `λₙ::AbstractFloat`: ???
- `λₗ::AbstractFloat`: ???

# Developers Notes
This function and its doc-string will be rewritten once its function and arguments have
been identified satisfactorily.

# Examples
This is primarily intended to act as a filter function for off site bases like so:   
```
julia> off_site_sym_basis = SymmetricBasis(
           φ, basis, selector,
           filterfun = bb -> filter_offsite_be(bb, deg)
```

# Todo
    - This should be rewritten to be cleaner and more performant.
"""
 function filter_offsite_be(bb, deg::Integer, λₙ::AbstractFloat=0.5, λₗ::AbstractFloat=0.5)
    # Issue a warning when this function is called until the issues with this function
    # have been resolved! 
    @warn "`filter_offsite_be` is not production ready!" maxlog=1
    if length(bb) == 0; return false; end 
    deg_n, deg_l = ceil(deg * λₙ), ceil(deg * λₗ)
    for b in bb
       if (b.be == :env) && (b.n>deg_n || b.l>deg_l)
          return false
       end
    end
    return ( sum( b.be == :bond for b in bb ) == 1 )
 end

end