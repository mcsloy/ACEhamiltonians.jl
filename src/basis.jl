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
    - If two or more basis functions have the same parameters and represent interactions
      between identical azimuthal pairs then they should share the same `SymmetricBasis`
      object. This will reduce memory and database size and will improve performance.
      However, the read/write_dict functions for the `Model` structure will need some
      subroutines to smartly (re)store bases with `SymmetricBasis` objects. The same should
      be true of the State information of like atoms.
    - Add show method do the basis classes.
    - Figure out what is going on with filter_offsite_be and its arguments.
    - A warning should perhaps be given if no filter function is given when one is
      expected; such as off-site functions. If no-filter function is desired than
      a dummy filter should be required.
    - Might be worth adding descriptive info to the bases; such as what shell does
      this correspond to, what azimuthal number, etc. This could be optional.
    - Add ison method for the basis entities.
    - Improve typing for the Model structure.
    - A method for storing tuples as keys in json needs to be created.
    - parse_key should be abstracted.
"""

"""
    Basis(basis[, coefficients, mean])



# Attributes:
- `basis::SymmetricBasis`: pass
- `coefficients::Array`: pass
- `mean::Real`: pass

# Constructors
Unfitted `Basis` instances can be instantiated either by providing a `SymmetricBasis`
object or using one of the convenience functions: [`on_site_basis`](@ref): for on_site
`Basis` instances and [`off_site_basis`](@ref): for off-site instances. However, these
produce simple `Basis` instances with mostly hard-coded parameters. If more fine-grained
control over the basis' parameters is required then `SymmetricBasis` instances will need
to be instantiated manually. Fully fitted `Basis` instances are constructed by specifying
all requisite attributes.


Todo:
    - Describe how they can be fitted (once the functions are in place)
    - Add a catch to the predictor to disallow predictions to be be made using
      basis instances that have not yet be fitted.
    - Check if `mean` is a scalar or vector and update read_dict as needed.


"""
mutable struct Basis
    basis::S where S<:SymmetricBasis
    id::T where T<:Tuple
    coefficients::Union{Array, Nothing}
    mean::Union{Array, Nothing}

    function Basis(basis, id)
        new(basis, id, nothing, nothing)
    end

    function Basis(basis, id, coefficients, mean)
        new(basis, id, coefficients, mean)
    end

end

Base.:(==)(x::Basis, y::Basis) = (
    x.id == y.id && x.coefficients == y.coefficients
    && x.mean == y.mean && x.basis == y.basis)

    
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

# ACEbase.write_dict(b::Basis) = Dict(
#     "__id__"=>"HBasis",
#     "basis"=>write_dict(b.basis),
#     "id"=>b.id,
#     "coefficients"=>b.coefficients,
#     "mean"=>b.mean)

# ACEbase.read_dict(::Val{:HBasis}, dict::Dict)::Basis = Basis(
#     read_dict(dict["basis"]),
#     Tuple(dict["id"]),
#     isnothing(dict["coefficients"]) ? nothing : Vector{Float64}(dict["coefficients"]),
#     isnothing(dict["mean"]) ? nothing : Vector{Float64}(dict["mean"])
# )



# Todo: this is mostly to stop terminal spam but needs to be updated
#       with more meaningful information later on.
function Base.show(io::IO, basis::Basis)
    print(io, "Basis(fitted=$(~isnothing(basis.mean)))")
end


"""Returns a boolean indicating if the basis instance represents an on-site interaction."""
Parameters.ison(basis::Basis) = length(basis.id) == 3

"""
_filter_bases(basis, type)

Helper function to retrieve specific basis function information out of a
`Basis` instance.

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

# on_site_models should be renamed to on_site_bases
struct Model
    # on_site_models::Dict{NTuple{4, I}, Basis} where I<:Integer  # z₁-z₂-ℓ₁-ℓ₂
    # off_site_models::Dict{NTuple{3, I}, Basis}  where I<:Integer # z-ℓ₁-ℓ₂
    # on_site_parameters::Parameters
    # off_site_parameters::Parameters

    on_site_models
    off_site_models
    on_site_parameters
    off_site_parameters

    function Model(on_site_models, off_site_models,
        on_site_parameters::ParaDef, off_site_parameters::ParaDef)
        new(on_site_models, off_site_models, on_site_parameters, off_site_parameters)
    end
    
    function Model(basis_definition::BasisDef, on_site_parameters::ParaDef,
                   off_site_parameters::ParaDef)
        # Developers Notes
        # This makes the assumption that all z₁-z₂-ℓ₁-ℓ₂ interactions are represented
        # by the same model.

        on_sites = Dict{NTuple{3, keytype(basis_definition)}, Basis}()
        off_sites = Dict{NTuple{4, keytype(basis_definition)}, Basis}()
        
        # Sorting the basis definition makes avoiding interaction doubling easier.
        # That is to say, we don't create models for both H-C and C-H interactions
        # as they represent the same thing.
        basis_definition_sorted = sort(collect(basis_definition), by=first) 

        # Loop over all species defined in the basis definition
        for (zₙ, (zᵢ, shellsᵢ)) in enumerate(basis_definition_sorted)
            # Construct on-site models for each unique shell pair (note ℓᵢ-ℓⱼ = ℓⱼ-ℓᵢ)
            for (n₁, ℓ₁) in enumerate(shellsᵢ)
                for (n₂, ℓ₂) in enumerate(shellsᵢ[n₁:end])
                    # Helper function reduce code repetition
                    f = i -> getfield(on_site_parameters, i)[zᵢ][n₁, n₂+n₁-1]

                    ace_basis = on_site_ace_basis(ℓ₁, ℓ₂,
                        f.((:ν, :deg, :e_cutₒᵤₜ, :e_cutᵢₙ))...)
                    
                    on_sites[(zᵢ, n₁, n₂+n₁-1)] = Basis(ace_basis, (zᵢ, n₁, n₂+n₁-1))

                end
            end

            # Loop over all possible unique species pair combinations
            for (zⱼ, shellsⱼ) in basis_definition_sorted[zₙ:end]
    
                # Construct the off site-models
                for (n₁, ℓ₁) in enumerate(shellsᵢ), (n₂, ℓ₂) in enumerate(shellsⱼ)
                    
                    # Ignore symmetrically equivalent homoatomic interactions; i.e. as
                    # Cₛ-Cₚ and Cₚ-Cₛ are the same interaction only one model is needed.
                    if zᵢ == zⱼ && n₁ > n₂
                        continue
                    end

                    f = i -> getfield(off_site_parameters, i)[(zᵢ, zⱼ)][n₁, n₂]

                    ace_basis = off_site_ace_basis(ℓ₁, ℓ₂,
                        f.((:ν, :deg, :e_cutₒᵤₜ, :e_cutᵢₙ, :bond_cut, :λₙ, :λₗ))...)

                    off_sites[(zᵢ, zⱼ, n₁, n₂)] = Basis(ace_basis, (zᵢ, zⱼ, n₁, n₂))
                end
            end

        end

    new(on_sites, off_sites, on_site_parameters, off_site_parameters)
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

Base.:(==)(x::Model, y::Model) = (
    x.on_site_models == y.on_site_models && x.off_site_models == y.off_site_models
    && x.on_site_parameters == y.on_site_parameters && x.off_site_parameters == y.off_site_parameters)

ACEbase.write_dict(m::Model) = Dict(
    "__id__"=>"HModel",
    "on_site_models"=>Dict(k=>write_dict(v) for (k, v) in m.on_site_models),
    "off_site_models"=>Dict(k=>write_dict(v) for (k, v) in m.off_site_models),
    "on_site_parameters"=>write_dict(m.on_site_parameters),
    "off_site_parameters"=>write_dict(m.off_site_parameters))

ACEbase.read_dict(::Val{:HModel}, dict::Dict)::Model = Model(
    Dict(parse_key(k)=>read_dict(v) for (k, v) in dict["on_site_models"]),
    Dict(parse_key(k)=>read_dict(v) for (k, v) in dict["off_site_models"]),
    read_dict(dict["on_site_parameters"]),
    read_dict(dict["off_site_parameters"]))


# Todo: this is mostly to stop terminal spam but needs to be updated
#       with more meaningful information later on.
function Base.show(io::IO, model::Model)

    # Work out if the on/off site models are full, Partially or un-fitted.
    f = b -> if all(b) "no" elseif all(!, b) "yes" else "partially" end
    on = f([isnothing(i.mean) for i in values(model.on_site_models)])
    off = f([isnothing(i.mean) for i in values(model.off_site_models)])
    
    # Identify the species present
    species = join(sort(unique(getindex.(collect(keys(model.on_site_models)), 1))), ", ", " & ")

    print(io, "Model(fitted=(on=$on, off=$off), species=($species))")
end


@doc raw"""
###### REWRITE DOCSTRING AS THIS NOW RETURNS AN ACE BASE

    on_site_basis(ℓ₁, ℓ₂, ν, deg, e_cutₒᵤₜ[, e_cutᵢₙ])

Initialise a simple on-site `Basis` instance with sensible default parameters.

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


# Todo
- add return documentation
- as many simplifications where made here there is a real chance that the basis
  generated by this method might return gibberish. So tolerance checks must be
  performed before this can be considered operational.
- complete the returns selection
- give examples
- discuss sensible choices for the parameters

# Developers Notes
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
###### REWRITE DOCSTRING AS THIS NOW RETURNS AN ACE BASE

    off_site_basis(ℓ₁, ℓ₂, ν, deg, b_cut[,e_cutₒᵤₜ, e_cutᵢₙ, λₙ, λₗ])


Initialise a simple off-site `Basis` instance with sensible default parameters.

Operates similarly to [`on_site_basis`](@ref) but applies a `CylindricalBondEnvelope` to
the `Rn1pBasis` basis instance. The length and radius of the cylinder are defined as
maths: ``b_{cut}+2r_{cut}`` and maths: ``r_{cut}`` respectively; all other parameters
resolve to their defaults as defined by their constructors. Again, `Bases` instances must
be manually instantiated if more fine-grained control is desired.

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

# Todo:
- many simplifications were made; thus there is no guarantee that the basis generated here
  are in any way meaningful.
- Identify what the filter function is doing and documenting it.

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