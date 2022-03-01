using HDF5
using ACE
using ACE: SymmetricBasis, SphericalMatrix, Utils.RnYlm_1pbasis, SimpleSparseBasis,
           CylindricalBondEnvelope, Categorical1pBasis, cutoff_radialbasis

using ACEhamiltonians
using ACEhamiltonians.Parameters
"""
TODO:
    - Add show method do the basis classes.
    - Figure out what is going on with filter_offsite_be and its arguments.
    - A warning should perhaps be given if no filter function is given when one is
      expected; such as off-site functions. If no-filter function is desired than
      a dummy filter should be required.
    - Might be worth adding descriptive info to the bases; such as what shell does
      this correspond to, what azimuthal number, etc. This could be optional.
    - Add ison method for the basis entities.
"""

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


"""
mutable struct Basis
    basis::SymmetricBasis
    coefficients::Union{Array, Nothing}
    mean::Union{Real, Nothing}

    function Basis(basis)
        new(basis, nothing, nothing)
    end
end

# Todo: this is mostly to stop terminal spam but needs to be updated
#       with more meaningful information later on.
function Base.show(io::IO, basis::Basis)
    print(io, "Basis(fitted=$(~isnothing(basis.mean)))")
end



struct Model
    # on_site_models::Dict{NTuple{4, I}, Basis} where I<:Integer  # z₁-z₂-ℓ₁-ℓ₂
    # off_site_models::Dict{NTuple{3, I}, Basis}  where I<:Integer # z-ℓ₁-ℓ₂
    # on_site_parameters::Parameters
    # off_site_parameters::Parameters

    on_site_models
    off_site_models
    on_site_parameters
    off_site_parameters
    
    function Model(basis_definition::BasisDef, on_site_parameters::ParaDef,
                   off_site_parameters::ParaDef)
        # Developers Notes
        # This makes the assumption that all z₁-z₂-ℓ₁-ℓ₂ interactions are represented
        # by the same model.

        on_sites = Dict{NTuple{3, keytype(basis_definition)}, Basis}()
        off_sites = Dict{NTuple{4, keytype(basis_definition)}, Basis}()
        

        # Loop over all species defined in the basis definition
        for (zᵢ, shellsᵢ) in basis_definition
            # Construct on-site models for each unique shell pair (note ℓᵢ-ℓⱼ = ℓⱼ-ℓᵢ)
            for (n₁, ℓ₁) in enumerate(shellsᵢ)
                for (n₂, ℓ₂) in enumerate(shellsᵢ[n₁:end])
                    # Helper function reduce code repetition
                    f = i -> getfield(on_site_parameters, i)[zᵢ][n₁, n₂+n₁-1]
                    on_sites[(zᵢ, ℓ₁, ℓ₂)] = on_site_basis(ℓ₁, ℓ₂,
                        f.((:ν, :deg, :e_cutₒᵤₜ, :e_cutᵢₙ))...)
                end
            end

            # Loop over all possible pairwise species combinations
            for (zⱼ, shellsⱼ) in basis_definition
            
                # Construct the off site-models
                for (n₁, ℓ₁) in enumerate(shellsᵢ), (n₂, ℓ₂) in enumerate(shellsⱼ)
                    # The off site keys (zᵢ,zⱼ) & (zⱼ,zᵢ) represent the same interaction;
                    # thus only one is present in off_site_parameters. As the lowest
                    # atomic number is given first the key must be sorted.
                    f₁ = zᵢ <= zⱼ ? (zᵢ, zⱼ) : (zⱼ, zᵢ)
                    f₂ = zᵢ <= zⱼ ? [n₁, n₂] : [n₂, n₁]
                    f = i -> getfield(off_site_parameters, i)[f₁][f₂...]
        
                    off_sites[(zᵢ, zⱼ, ℓ₁, ℓ₂)] = off_site_basis(ℓ₁, ℓ₂,
                        f.((:ν, :deg, :e_cutₒᵤₜ, :e_cutᵢₙ, :bond_cut, :λₙ, :λₗ))...)
                end
            end

        end

    new(on_sites, off_sites, on_site_parameters, off_site_parameters)
    end

    function Model(basis_definition::BasisDef)
        parameters = ParaDef(basis_definition)
        Model(basis_definition, parameters)
        
    end

    function Model(source::HDF5.Group)
        # Load model from HDF5 group
        error("Not implemented")
    end

end

@doc raw"""

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
pass

# Todo
- as many simplifications where made here there is a real chance that the basis
  generated by this method might return gibberish. So tolerance checks must be
  performed before this can be considered operational.
- complete the returns selection
- give examples
- discuss sensible choices for the parameters

# Developers Notes
"""
function on_site_basis(ℓ₁::I, ℓ₂::I, ν::I, deg::I, e_cutₒᵤₜ::F, e_cutᵢₙ::F=2.5
    ) where {I<:Integer, F<:AbstractFloat}
    # Build i) a matrix indicating the desired sub-block shape, ii) the one
    # particle Rₙ·Yₗᵐ basis describing the environment, & iii) the basis selector.
    # Then instantiate the SymmetricBasis required by the Basis structure.
    return Basis(SymmetricBasis(
        SphericalMatrix(ℓ₁, ℓ₂; T=ComplexF64),
        RnYlm_1pbasis(maxdeg=deg, r0=e_cutᵢₙ, rcut=e_cutₒᵤₜ),
        SimpleSparseBasis(ν, deg)))
end

@doc raw"""

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
function off_site_basis(ℓ₁::I, ℓ₂::I, ν::I, deg::I, b_cut::F, e_cutₒᵤₜ::F=5., e_cutᵢₙ::F=2.5,
    λₙ::F=.5, λₗ::F=.5) where {I<:Integer, F<:AbstractFloat}

    # Bond envelope which controls which atoms are seen by the bond.
    env = CylindricalBondEnvelope(b_cut, e_cutₒᵤₜ, e_cutₒᵤₜ)

    # Categorical1pBasis is applied to the basis to allow atoms which are part of the
    # bond to be treated differently to those that are just part of the environment.
    discriminator = Categorical1pBasis([:bond, :env]; varsym=:be, idxsym=:be)

    # The basis upon which the above entities act.
    RnYlm = RnYlm_1pbasis(maxdeg=deg, r0=e_cutᵢₙ, rcut=cutoff_radialbasis(env))
    
    return Basis(SymmetricBasis(
        SphericalMatrix(ℓ₁, ℓ₂; T=ComplexF64),
        RnYlm * env * discriminator,
        SimpleSparseBasis(ν + 1, deg),
        filterfun=bb -> filter_offsite_be(bb, deg, λₙ, λₗ)))
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




# BasisDef = Dict{Integer, Vector{Integer}}

# struct Model
#     on_site_models::Dict{NTuple{4, Integer}, Basis}  # z₁-z₂-ℓ₁-ℓ₂
#     off_site_models::Dict{NTuple{3, Integer}, Basis} # z-ℓ₁-ℓ₂
#     on_site_parameters::Parameters
#     off_site_parameters::Parameters
    
#     function Model(basis_definition::BasisDef, on_site_parameters::ParaDef,
#                    off_site_parameters::ParaDef)

basis_def = Dict(1=>[0], 6=>[0,0,1])
pd_on = ParaDef(basis_def, 2, 8, site="on")
pd_off = ParaDef(basis_def, 2, 8, site="off")
Model(basis_def, pd_on, pd_off);

# v, 6, )

# get.(getfield((pd_on,), (:ν, :deg, :e_cutₒᵤₜ, :e_cutᵢₙ)), 6, nothing)