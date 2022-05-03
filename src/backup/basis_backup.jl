module Bases
# using HDF5
using HDF5: Group
using ACE
using ACE: SymmetricBasis, SphericalMatrix, Utils.RnYlm_1pbasis, SimpleSparseBasis,
           CylindricalBondEnvelope, Categorical1pBasis, cutoff_radialbasis
import ACEbase
using ACEhamiltonians
# using ACEhamiltonians.Parameters


export Basis, BasisDef, radial, angular, categorical, envelope, Model, on_site_ace_basis, off_site_ace_basis, filter_offsite_be
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

"""

    Basis(basis, id[, coefficients, mean])

Structure responsible for holding and allowing for ease of manipulation of ACE bases.


# Attributes:
- `basis::SymmetricBasis`: The ACE basis object.  
- `id::Tuple': A tuple identifying the interaction to which the basis pertains. This should
    take the forms (Zᵢ, sᵢ, sⱼ) & (Zᵢ, Zⱼ, sᵢ, sⱼ) for on and off-site bases respectively.
    Where Zᵢ/Zⱼ are the atomic numbers and and sᵢ/sⱼ are the shell numbers. 
- `coefficients::Vector{T<:AbstractReal}`: ACE basis coefficients as determined by the
    least squares fit.
- `mean::Matrix{T<:AbstractReal}`: Mean values (offset) as obtained during the least
    squares fitting process.

# Constructors
Unfitted `Basis` instances can be instantiated either by providing a `SymmetricBasis`
object, which can be constructed with one of the convenience functions:
[`on_site_ace_basis`](@ref): for on-site instances and [`off_site_ace_basis`](@ref): for
off-site instances. However, these produce simple `SymmetricBasis` instances with mostly
hard-coded parameters. If more fine-grained control over the basis' parameters is required
then `SymmetricBasis` instances will need to be instantiated manually. Fully fitted
`Basis` instances are constructed by specifying all requisite attributes.

# Notes
`Basis` instances can be fitted using one of the various `fit!` methods present in the
`ACEhamiltonians.Fitting` module.

It is worth remarking that while the underling ACE `SymmetricBasis` entity is a complex
type, as defined by its value type `SphericalMatrix`, the actual results returned by
the calculation at the end are real. Thus the type here is always real.


Todo:
    - Add a catch to the predictor to disallow predictions to be be made using
      basis instances that have not yet be fitted.
    - Check if `mean` is a scalar or vector and update read_dict as needed.


"""
mutable struct Basis{T<:AbstractFloat}
    basis::S where S<:SymmetricBasis
    id::T where T<:Tuple
    coefficients::Union{Vector{T}, Nothing}
    mean::Union{Matrix{T}, Nothing}

    function Basis(basis, id)
        # Even though SphericalMatrix may say it is complex the results will always be
        # real, thus the type gets down-converted to its real equivalent.
        type = real(ACE.valtype(basis).parameters[5])
        new{type}(basis, id, nothing, nothing)
    end

    function Basis(basis, id, coefficients, mean)
        type = real(ACE.valtype(basis).parameters[5])
        new{type}(basis, id, coefficients, mean)
    end

end

# Associated helper methods

Base.:(==)(x::Basis, y::Basis) = (
    x.id == y.id && x.coefficients == y.coefficients
    && x.mean == y.mean && x.basis == y.basis)

function Base.show(io::IO, basis::Basis)
    print(io, "Basis(id: $(basis.id), fitted: $(~isnothing(basis.mean)))")
end

"""Expected shape of the sub-block associated with the `Basis`; 3×3 for a pp basis etc."""
Base.size(basis::Basis) = (ACE.valtype(basis.basis).parameters[3:4]...,)

"""Expected type of resulting sub-blocks."""
Base.valtype(::Basis{T}) where T = T

"""Azimuthal quantum numbers associated with the `Basis`."""
azimuthals(basis::Basis) = (ACE.valtype(basis.basis).parameters[1:2]...,)


function ACEbase.write_dict(b::Basis)
    # Handle parsing of entities which may be arrays or `nothing`
    p(i) = isnothing(i) ? i : write_dict(i)
    return Dict(
    "__id__"=>"HBasis",
    "basis"=>write_dict(b.basis),
    "id"=>b.id,
    "coefficients"=>p(b.coefficients),
    "mean"=>p(b.mean))
end

function ACEbase.read_dict(::Val{:HBasis}, dict::Dict)::Basis
    # Handle parsing of entities which may be arrays or `nothing`
    p(i) = isnothing(i) ? i : read_dict(i)
    return Basis(
        read_dict(dict["basis"]),
        Tuple(dict["id"]),
        p(dict["coefficients"]),
        p(dict["mean"]))
end


"""Returns a boolean indicating if the basis instance represents an on-site interaction."""
Parameters.ison(basis::Basis) = length(basis.id) == 3

"""
_filter_bases(basis, type)

Helper function to retrieve specific basis function information out of a `Basis` instance.

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

###################
# Model Structure #
###################
struct Model
    # on_site_models::Dict{NTuple{4, I}, Basis} where I<:Integer  # z₁-z₂-ℓ₁-ℓ₂
    # off_site_models::Dict{NTuple{3, I}, Basis}  where I<:Integer # z-ℓ₁-ℓ₂
    # on_site_parameters::Parameters
    # off_site_parameters::Parameters

    # on_site_models
    # off_site_models

    on_site_bases
    off_site_bases

    on_site_parameters
    off_site_parameters
    basis_definition

    function Model(on_site_bases, off_site_bases,
        on_site_parameters::ParaDef, off_site_parameters::ParaDef, basis_definition)
        new(on_site_bases, off_site_bases, on_site_parameters, off_site_parameters, basis_definition)
    end
    
    function Model(basis_definition::BasisDef, on_site_parameters::ParaDef,
                   off_site_parameters::ParaDef)
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
                for (n₁, ℓ₁) in enumerate(shellsᵢ), (n₂, ℓ₂) in enumerate(shellsⱼ)
                    # Skip symmetrically equivalent interactions. 
                    zᵢ == zⱼ && n₁ > n₂ && continue
                    
                    if zᵢ == zⱼ 
                        ace_basis = ace_basis_on( # On-site bases
                            ℓ₁, ℓ₂, gather(on_site_parameters, zᵢ, n₁, n₂)...)
                        on_sites[(zᵢ, n₁, n₂)] = Basis(ace_basis, (zᵢ, n₁, n₂))
                    end

                    ace_basis = ace_basis_off( # Off-site bases
                        ℓ₁, ℓ₂, gather(off_site_parameters, zᵢ, zⱼ, n₁, n₂)...)

                    off_sites[(zᵢ, zⱼ, n₁, n₂)] = Basis(ace_basis, (zᵢ, zⱼ, n₁, n₂))
                end
            end
        end

    new(on_sites, off_sites, on_site_parameters, off_site_parameters, basis_definition)
    end

    # function Model(basis_definition::BasisDef)
    #     # Double check if this is really needed, document if so remove if not.
    #     parameters = ParaDef(basis_definition)
    #     return Model(basis_definition, parameters)
    # end

    # function Model(source::HDF5.Group)
    #     # Load model from HDF5 group
    #     error("Not implemented")
    # end

end

# Associated methods

Base.:(==)(x::Model, y::Model) = (
    x.on_site_bases == y.on_site_bases && x.off_site_bases == y.off_site_bases
    && x.on_site_parameters == y.on_site_parameters && x.off_site_parameters == y.off_site_parameters)

function ACEbase.write_dict(m::Model)
    # ACE bases are stored as hash values which are checked against the "bases_hashes"
    # dictionary during reading. This avoids saving multiple copies of the same object;
    # which is common as `Basis` objects tend to share basis functions.
    dict =  Dict(
        "__id__"=>"HModel",
        "on_site_bases"=>Dict(k=>write_dict(v) for (k, v) in m.on_site_bases),
        "off_site_bases"=>Dict(k=>write_dict(v) for (k, v) in m.off_site_bases),
        "on_site_parameters"=>write_dict(m.on_site_parameters),
        "off_site_parameters"=>write_dict(m.off_site_parameters),
        "basis_definition"=>Dict(k=>write_dict(v) for (k, v) in m.basis_definition),
        "bases_hashes"=>merge(
            Dict(string(hash(m.basis))=>write_dict(m.basis) for m in values(m.on_site_bases)),
            Dict(string(hash(m.basis))=>write_dict(m.basis) for m in values(m.off_site_bases))))
    
    # Replace the bases' basis objects with a hash
    for (k, v) in dict["on_site_bases"]
        v["basis"] = string(hash(m.on_site_bases[k].basis))
    end
    for (k, v) in dict["off_site_bases"]
        v["basis"] = string(hash(m.off_site_bases[k].basis))
    end

    return dict
end


##########################
##########################
# NEED TO UPDATE CODE SO THAT MODELS ARE CALLED BASES
##########################
##########################
function ACEbase.read_dict(::Val{:HModel}, dict::Dict)::Model

    # Construct basis function loop-up dictionary
    basis_functions = Dict(k=>read_dict(v) for (k, v) in dict["bases_hashes"])
    ensure_int(v) = v isa String ? parse(Int, v) : v
    # Regenerate basis function
    regen_basis(v) = Basis(
        basis_functions[v["basis"]],
        Tuple(v["id"]),
        isnothing(v["coefficients"]) ? nothing : read_dict(v["coefficients"]),
        isnothing(v["mean"]) ? nothing : read_dict(v["mean"]))

    return Model(
        Dict(parse_key(k)=>regen_basis(v) for (k, v) in dict["on_site_bases"]),
        Dict(parse_key(k)=>regen_basis(v) for (k, v) in dict["off_site_bases"]),
        read_dict(dict["on_site_parameters"]),
        read_dict(dict["off_site_parameters"]),
        Dict(ensure_int(k)=>read_dict(v) for (k, v) in dict["basis_definition"]))
end



# Todo: this is mostly to stop terminal spam but needs to be updated
#       with more meaningful information later on.
function Base.show(io::IO, model::Model)

    # Work out if the on/off site bases are fully, partially or un-fitted.
    f = b -> if all(b) "no" elseif all(!, b) "yes" else "partially" end
    on = f([isnothing(i.mean) for i in values(model.on_site_bases)])
    off = f([isnothing(i.mean) for i in values(model.off_site_bases)])
    
    # Identify the species present
    species = join(sort(unique(getindex.(collect(keys(model.on_site_bases)), 1))), ", ", " & ")

    print(io, "Model(fitted=(on: $on, off: $off), species: ($species))")
end


##########################
# ACE Basis Constructors #
##########################


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
function off_site_ace_basis(ℓ₁::I, ℓ₂::I, ν::I, deg::I, b_cut::F, e_cutₒᵤₜ::F=5., e_cutᵢₙ::F=2.5,
    λₙ::F=.5, λₗ::F=.5) where {I<:Integer, F<:AbstractFloat}

    # Bond envelope which controls which atoms are seen by the bond.
    env = CylindricalBondEnvelope(b_cut, e_cutₒᵤₜ, e_cutₒᵤₜ)

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


# Todo: Remove this temporary function
function off_site_ace_basis(ℓ₁::I, ℓ₂::I, ν::I, deg::I, env::CylindricalBondEnvelope,
    e_cutᵢₙ::F=2.5, λₙ::F=.5, λₗ::F=.5) where {I<:Integer, F<:AbstractFloat}

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