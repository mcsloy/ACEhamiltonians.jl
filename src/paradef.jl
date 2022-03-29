module Parameters
import ACEbase.FIO: read_dict, write_dict
import ACEbase
using ACEhamiltonians

export ParaDef, show, ison, read_dict, write_dict, gather

# Todo:
#   - Refactor read/write_dict methods to make use of vector and matrix
#     conversion methods already present in ACEbase. 

# Type definition used to simplify ParaDef construction
TP = Dict{K, Matrix{V}} where {K, V}
P = Union{R, Dict{K, A}} where {K<:Union{Tuple{I, I}, R}, A<:Any} where {R<:Real, I<:Integer}

# Todo: Move this elsewhere
BasisDef = Dict{I, Vector{I}} where I<:Integer

"""
    ParaDef(ν, deg, e_cutₒᵤₜ, e_cutᵢₙ[, bond_cut, λₙ, λₗ])

``ParaDef`` instances specify the parameters to be used when constructing the on/off-site
models. Each parameter is represented by a dictionary with one entry per-species, or per-
species-pair in the off-site case. The value of each entry is a matrix whose elements are
the parameters of the associated shell-shell interactions. The form of these dictionaries
and their keys/values is discussed in more detail in the "Argument Structure" section.

While parameter definitions can be created manually it is often more convenient to use the
constructor, as it accepts contracted argument forms. Meaning large parameter definitions
can be created with comparative ease. See the "Constructor" section for more detail.

# Fields
- `ν`: maximum correlation order.
- `deg`: maximum polynomial degree.
- `e_cutₒᵤₜ`: outer environmental cutoff distance.  
- `e_cutᵢₙ`: inner environmental cutoff distance.
- `bond_cut`: bond cutoff distances (off-site-only)
- `λₙ`: ??? (off-site only) TODO: expand
- `λₗ`: ??? (off-site only) TODO: expand

# Argument Structure
## On-site Parameters
On-site parameters are passed via dictionaries which are keyed by atomic number and valued
by two dimensional matrices containing one value per shell pair. An arbitrary parameter "p"
would generally be described like so:
```
p = Dict(
  Zᵢ => [pⁱ₁₁ pⁱ₁₂ ⋯ pⁱ₁ₙ
        pⁱ₂₁ pⁱ₂₂ ⋯ pⁱ₂ₙ
         ⋮   ⋮  ⋱  ⋮ 
        pⁱₙ₁ pⁱₙ₂ ⋯ pⁱₙₙ],

  Zⱼ => [pʲ₁₁ pʲ₁₂ ⋯ pʲ₁ₙ
        pʲ₂₁ pʲ₂₂ ⋯ pʲ₂ₙ
         ⋮   ⋮  ⋱  ⋮ 
        pʲₙ₁ pʲₙ₂ ⋯ pʲₙₙ],
  ⋯
)
```
Where ``Zᵢ`` is the atomic numbers of species "i" and ``pⁱₘₙ`` is the parameter associated
with the model representing the on-site interaction between the mᵗʰ & nᵗʰ shells.

There should be one entry in the dictionary for each unique species type and one value for
each shell pair; thus the call ``p[i][m,n]`` would yield the parameter to be used by the
on-site model representing the interactions between the mᵗʰ and nᵗʰ shells on species "i".
It is important to note that: i) the order in which shells are specified must match the
order in which they appear in the basis definition, and ii) on-site interactions are
symmetric, thus the matrix ``pⁱ`` must also be symmetric, i.e. ``pⁱₙₘ`` and ``pⁱₘₙ`` must
be equivalent. For example, the maximum correlation order, ν, of a hydrocarbon system
with minimal basis would be specified as:
```
ν = Dict(1 =>[[2]], 6 =>[3 4 4; 4 5 5; 4 5 5])
```
Thus a maximum correlation order of 2 would be used for hydrogen ss interactions and 
values of 3, 4, & 5 would be used or all carbon ss, sp, & pp interactions respectively.

## Off-site Parameters
Off-site parameters are specified in a similar manner to their on-site counterparts, but
with two important differences. Firstly, keys must be tuples rather than integers as they
now correspond to interactions between pairs of species. Secondly, value matrices are no
longer required be to be square or symmetric. Off-site Parameter arguments take the
general form:
```
p = Dict(
  (Zᵢ,Zⱼ) => [pⁱʲ₁₁ pⁱʲ₁₂ ⋯ pⁱʲ₁ₙ
            pⁱʲ₂₁ pⁱʲ₂₂ ⋯ pⁱʲ₂ₙ
             ⋮   ⋮  ⋱  ⋮ 
            pⁱʲₘ₁ pⁱʲₘ₂ ⋯ pⁱʲₘₙ],
  ⋯
)

```
Where the key ``(Zₘ, Zₙ)`` specifies the atomic numbers of the species pair to which its
parameters pertain; and``pⁱʲₘ`` is the value associated with the interaction between
shells "m" and "n" on species "i" and "j" respectively. Again, the order in which the
values are specified in the matrix must match up with the order of the basis definition.
Finally, each species pair should appear only once i.e. if the key ``(Zᵢ,Zⱼ)`` is present
then the key ``(Zⱼ,Zᵢ)`` should not be, as they both represent the same interaction.

## Warnings
The mixing of on-site and off-site parameters is strictly forbidden. A separate `ParaDef`
instance is required for each. As the on-site interactions are symmetric their parameter
matrices must also be. While this restriction does not apply to the off-site interactions
it is ill-advised to have non-symmetric parameter matrices for homo-atomic interactions.

Off-site interaction keys will be sorted internally; i.e. (6, 1) will be auto-converted
into (1, 6).


## Developers Notes
This will be refactored to account for the fact that homo-atomic off-site parameters must
be symmetric.
"""
mutable struct ParaDef{Ti,Tf}
    ν::Ti
    deg::Ti
    e_cutₒᵤₜ::Tf
    e_cutᵢₙ::Tf
    bond_cut::Union{Tf,Nothing}
    λₙ::Union{Tf,Nothing}
    λₗ::Union{Tf,Nothing}
    

    """Default off-site constructor for manual initialisation"""
    function ParaDef(ν::Ti, deg::Ti, e_cutₒᵤₜ::Tf, e_cutᵢₙ::Tf, bond_cut::Tf, λₙ::Tf, λₗ::Tf
        ) where {Ti<:TP{Tuple{I,I}, I}, Tf<:TP{Tuple{I,I}, F}
        } where {I<:Integer, F<:AbstractFloat}
        # Gather arguments together to simplify manipulation
        args = (ν, deg, e_cutₒᵤₜ, e_cutᵢₙ, bond_cut, λₙ, λₗ)

        # Ensure keys are atomic number minor, i.e. the lowest atomic number is the first
        # entry in each tuple; e.g. (H-C) not (C-H). This consistency helps simplify the
        # lower level code.
        sorted_args = (d -> Dict(issorted(k) ? k=>v : reverse(k)=>collect(v')
                          for (k, v) in d)).(args)

        # Ensure no doubly defined interactions exist; e.g. if C-H is present then H-C
        # must not be. The above key-sort causes redundant keys to vanish, e.g.
        # Dict((1,6)=>1,(6,1)=>2)→Dict((1,6)=>2). Thus, checking if the number of keys
        # present in each dictionary changes once the keys have been sorting is enough. 
        # 1) Get the length of each dictionary & its sorted counterpart.
        # 2) Ensure the length values within each tuple are the same.
        # 3) Raise an error if any dictionary's length is found to have changed.  
        if ~all(.==((i -> length.(keys.(i))).((args, sorted_args))...))
            error("Off-site dictionaries must not contain symmetrically equivalent keys; "*
                  "i.e. if key (1, 6) is present then (6, 6) cannot.")
        end

        # Parameter matrices are expected to be symmetric for homo-atomic species pairs.
        # The user should be warned if this is not the case.
        non_sym_keys = unique(vcat(collect.(keys.(filter.(
            p->~allunique(p.first) && p.second != p.second',  sorted_args)))...))

        if length(non_sym_keys) != 0
            @warn "Non-symmetric off-site parameter matrices are ill-advised for homoatomic"*
            " interactions: $(join(non_sym_keys, ", "))"
        end
        
        new{Ti, Tf}(sorted_args...)
    end

    """Default on-site constructor for manual initialisation"""
    function ParaDef(ν::Ti, deg::Ti, e_cutₒᵤₜ::Tf, e_cutᵢₙ::Tf
        ) where {Ti<:TP{I, I}, Tf<:TP{I, F}
        } where {I<:Integer, F<:AbstractFloat}


        # On-site parameter matrices must be symmetric; error out if any are not.
        non_sym_keys = unique(vcat(collect.(keys.(filter.(
            p->p.second != p.second',  (ν, deg, e_cutₒᵤₜ, e_cutᵢₙ))))...))

        if length(non_sym_keys) != 0
            error("Non-symmetric parameter matrices not permitted for on-site "*
                  "interactions: $(join(non_sym_keys, ", "))")
        end

        new{Ti, Tf}(ν, deg, e_cutₒᵤₜ, e_cutᵢₙ, nothing, nothing, nothing)
    end

    """Dictionary based constructor used to initialise ParaDef instances from Json files"""
    function ParaDef(dict)
        # Used to selectively skip off-site only parameters
        offset = (haskey(dict, "λₙ") && ~isnothing(dict["λₙ"])) ? 0 : 3
        ParaDef((dict["$i"] for i in fieldnames(ParaDef)[1:end-offset])...)        
    end

end

"""
    ParaDef(ν, deg, e_cutₒᵤₜ, e_cutᵢₙ[, bond_cut, λₙ, λₗ])

This enables lengthy parameter sets to be generated without having to manually construct
each full matrix. However, it is important to note that this just expands the contracted
definitions, as provided by the user, into their fully verbose forms. The constructor is
invoked whenever a basis definition is supplied as the first argument. In most instances
the (on/off-) site can be inferred from context. The exception to this is if only global
declinations are supplied, in which case the ``site`` keyword argument must be supplied
and set to either "on" or "off" as appropriate. There are four deceleration schemes that
are accepted by the constructor; global, species resolved, azimuthal resolved, and shell
resolved. Each of which is discussed in turn below.

Global assignment is invoked by providing a single value, rather than a dictionary, for a
parameter. This informs the constructor that this value should be used by all models. For
example, the maximum correlation order, ν, can be given as a single value "`2`":
```
julia> basis_def = Dict(1=>[0], 6=>[0,0,1])
julia> ν = 2
julia> parameters = ParaDef(basis_def, ν, 8, site="on")
julia> println(parameters.ν)
       Dict(6 => [2 2 2; 2 2 2; 2 2 2], 1 => [2;;])
```
It can be seen that this results in all on-site interactions using a common value.
Alternatively, parameters can be defined per-species, or per-species-pair for off-sites,
by providing a dictionary valued by a single scalar rather than a matrix:
```
julia> ν = Dict(1=>2, 6=>3)
julia> parameters = ParaDef(basis_def, ν, 8)
julia> println(parameters.ν)
       Dict(6 => [3 3 3; 3 3 3; 3 3 3], 1 => [2;;])
```
In the above example it can be seen that ν is "2" & "3" for all hydrogen & carbon on-site
interactions respectively. A finer degree of control over the parameters can be achieved
using azimuthal resolved decelerations. This allows for values to be specified not only
for each species but for each unique azimuthal pair on said species.
```
julia>#            Hss       Css Csp Cps Cpp
julia>#             ↓         ↓   ↓   ↓   ↓
julia> ν = Dict(1=>[2;;], 6=>[3   4;  4   5])
julia> parameters = ParaDef(basis_def, ν, 8)
julia> println(parameters.ν)
       #          s₁ s₂ p₁
       #          ↓ ↓ ↓
       Dict(6 => [3 3 4   # ← s₁
                  3 3 4   # ← s₂
                  4 4 5], # ← p₁
            1 => [2;;])
```
Here only a single value is provided for hydrogen as it has a single shell, whereas four
values are provided for carbon as it has two unique shell types (s & p) and thus four
possible unique interaction types (ss, sp, ps & pp). Finally, parameters can be specified
down to the shell level. However, this is no different than direct initialisation of a
``ParaDef`` instance via the base constructor. It is worth noting that decelerations can
be mixed; e.g.  `Dict(1=>[2;;], 6=>[3 4; 4 5]), 7=>6)`.

"""
function ParaDef(basis_def::BasisDef, ν::P, deg::P, e_cutₒᵤₜ::P=12.0, e_cutᵢₙ::P=2.0,
    bond_cut::Union{P, Nothing}=nothing, λₙ::Union{P, Nothing}=nothing,
    λₗ::Union{P, Nothing}=nothing; site::String="auto")

    # Ensure validity of the `site` argument
    @assert site in ["on", "off", "auto"] "Invalid site argument given (\"$site\"), valid"*
                                        " options are \"on\", \"off\" & \"auto\""

    # Collate common and off-site only arguments into a pair of tuples  
    c_args, off_args = (ν, deg, e_cutₒᵤₜ, e_cutᵢₙ), (bond_cut, λₙ, λₗ)

    # If site auto-detect enabled then infer the correct site. Auto-detection is always
    # possible so long as either one non-global argument is given or a off-site only
    # parameter is provided.
    if site == "auto"
        # If dictionaries are keyed by tuples; then this must be the off-site case. 
        if any([keytype(i)<:Tuple for i in c_args if i isa Dict])
            site = "off"

        # If all arguments are scalar com_args then it is not possible to auto-determine site
        elseif all([i isa Real for i in c_args])
            error("Site cannot be auto-resolve if all mutual arguments are specified"*
                  " globally; please provide the \"site\" keyword argument.")

        # If the site is not off-site, and is not non-inferable then it must be on-site
        else
            site = "on"
        end
    end

    # If λ terms are given for on-site parameters; issue a warning.
    if ~(isnothing(λₙ) & isnothing(λₗ) & isnothing(bond_cut)) && site=="on"
        @warn "Arguments bond_cut, λₙ & λₗ are off-site exclusive and will be ignored." 
    end

    # Create shorthand for _parameter_expander (reduces verbosity). The parameter
    # expander is used to expand short-form parameter decelerations as and if necessary.
    # Expand the λ terms if required; default them to 0.5 if unspecified. Finally,
    # instantiate the ParDef instance by redirecting to the base constructors.
    pex = var -> _parameter_expander(var, basis_def, site)
    if site == "off"
        off_args_exp = pex.((
            isnothing(λₙ) ? 12.0 : bond_cut,
            isnothing(λₙ) ? 0.5 : λₙ,
            isnothing(λₗ) ? 0.5 : λₗ))
    else
        off_args_exp = ()
    end
    ParaDef(pex.(c_args)..., off_args_exp...)

end



# Functions and methods associated with the ``ParaDef`` structure 

# These read and write functions allow for ParaDict⇄Dict interconversion. Such methods
# are most commonly used when writing to and from Json files.
# read_dict(::Val{:ParaDef}, dict::Dict)::ParaDef = ParaDef(dict)
# ACEbase.write_dict(p::ParaDef) = Dict("__id__"=>"ParaDef",
#                               ("$f"=>getfield(p, f) for f in fieldnames(ParaDef))...)

function ACEbase.write_dict(p::ParaDef)
    # Formats entity as a matrix if appropriate (null-op if v==nothing)
    from_mat(v) = isnothing(v) ? v : Dict(k=>write_dict(vv) for (k, vv) in v)
    return Dict(
        "__id__"=>"ParaDef",
        ("$f"=>from_mat(getfield(p, f)) for f in fieldnames(ParaDef))...)
end

function ACEbase.read_dict(::Val{:ParaDef}, dict::Dict)::ParaDef
    # Matrix objects are read from JSON files as Vector{Vector{Any}} instances. This
    # function is designed to convert such instances into matrices. 
    to_mat(v) = isnothing(v) ? v : read_dict(v)

    # Keys are stored a strings and must be converted back to integers
    format(data) = Dict(parse_key(k)=>to_mat(v) for (k, v) in data)

    on_site = isnothing(get(dict, "bond_cut", nothing)) 
    return ParaDef(Dict(
        "ν"=>format(dict["ν"]),
        "deg"=>format(dict["deg"]),
        "e_cutₒᵤₜ"=>format(dict["e_cutₒᵤₜ"]),
        "e_cutᵢₙ"=>format(dict["e_cutᵢₙ"]),
        "bond_cut"=>on_site ? nothing : format(dict["bond_cut"]),
        "λₙ"=>on_site ? nothing : format(dict["λₙ"]),
        "λₗ"=>on_site ? nothing : format(dict["λₗ"]),
    ))
end



function gather(para::ParaDef, z, i, j)
    # Todo: document this
    return (f -> getfield(para, f)[z][i, j]).(
        (:ν, :deg, :e_cutₒᵤₜ, :e_cutᵢₙ))
end

function gather(para::ParaDef, z_1, z_2, i, j)
    # Todo: document this
    return (f -> getfield(para, f)[(z_1, z_2)][i, j]).(
        (:ν, :deg, :e_cutₒᵤₜ, :e_cutᵢₙ, :bond_cut, :λₙ, :λₗ))
end    


"""
    ison(para_def)

Returns `true` if the parameter definition represents on-site parameters. 
"""
ison(para_def::ParaDef) = ~(keytype(para_def.ν)<:Tuple)
# Note "isoff" isn't implemented as it's just a negation of "ison" & is thus redundant. 

# Pretty printing implementation for the ParaDef structure. Without this julia will just
# vomit out vast amounts of data at the user.
function Base.show(io::IO, para_def::ParaDef)
    site = ison(para_def) ? "on" : "off"
    form = ison(para_def) ? "species" : "species-pairs"
    entries = join(join.(collect(keys(para_def.ν)), "-"), ", ")
    print(io, "ParaDef(site=$site, $form=($entries))")
end

function Base.:(==)(x::ParaDef, y::ParaDef)

    # Are the two instances are the same object?
    if x === y
        return true
    # Do they even reference the same site?
    elseif ison(x) != ison(y)
        return false
    end
    
    # Check all fields are identical
    for f in fieldnames(ParaDef)
        # If any do not match up then return false
        if getfield(x, f) != getfield(y, f)
            return false
        end
    end
    
    # If there are no difference they they must be equivalent
    return true

end

# Helper functions used by the `ParaDef` structure or its constructors

# Identifies unique vales in a vector and returns a series of vectors
# specifying the indices at which each unique element is found.
_unique_indices = vec -> map(k -> findall(vec .== k), unique(vec))

# Internal functions to help convert vectors/vectors of vectors to matrices
_vector_to_matrix(x::Vector{R}) where R<:Real = reshape(x, 1, length(x))
_vector_to_matrix(x::Vector{Vector{R}}) where R<:Real = mapreduce(permutedims, vcat, x)

function _parameter_expander(parameters::Union{Real, Dict{K, A}, Nothing}, basis_def::BasisDef,
    site::String) where {A<:Any, K<:Union{I, Tuple{I,I}}} where I<:Integer
    # This dispatch mostly acts as a proxy to i) ensure all function calls can be made
    # with an identical signature, and ii) to handel global parameter decelerations. 


    # Ensure validity of the `site` argument
    @assert site in ["on", "off", "auto"] "Invalid site argument given (\"$site\"), valid"*
                                          " options are \"on\", \"off\" & \"auto\""

    # Global parameter declarations are converted into their species resolved equivalents 
    # and are then fed back into the expander. Note, auto-site resolution is not possible
    # for global parameter declarations.
    if parameters isa Real
        species = collect(keys(basis_def))
        if site == "on"
            parameters = Dict(s => parameters for s in species)
        elseif site == "off"
            parameters = Dict((s₁<=s₂ ? (s₁, s₂) : (s₂, s₁))=> parameters
                              for (i, s₁) in enumerate(species)
                              for s₂ in species[i:end])
        else
            error("Site cannot be auto-resolve for global declarations.")
        end
    end

    # Now that any global site declarations have been down-converted to their species
    # resolved equivalents, the main expansion subroutines can now take over.
    # return _parameter_expander(parameters, basis_def)
    return _parameter_expander(parameters, basis_def)
end


function _parameter_expander(
    parameters::Dict{K, A}, basis_def::BasisDef) where {K<:Union{Tuple{I, I}, I}} where {I<:Integer, A<:Any}
    new_parameters, type = nothing, nothing

    for (i, (species, parameter)) in enumerate(parameters)

        # Ensure any vectors are up-converted to matrices; this makes life easier
        # from both a user and a developmental perspective.   
        if parameter isa Vector parameter = _vector_to_matrix(parameter) end

        # Initialise new_parameter dictionary & infer expected type; if "parameter" is a
        # matrix then the primitive type must be extracted via the .parameters attribute,
        if i == 1
            type = typeof(parameter)
            type = isprimitivetype(type) ? type : type.parameters[1]
            new_parameters = Dict{typeof(species), Matrix{type}}()
        end

        # Fetch the shell lists for the relevant species then work out i) the total number
        # of shells and ii) the number of different shell types on each species.
        shells = map(x -> basis_def[x], tuple(species...))[[1, end]]
        ns₁, ns₂ = length.(shells)
        nus₁, nus₂ = length.(unique.(shells))
         
        if parameter isa Real  # Species resolved declarations
            parameter = fill(parameter, (ns₁, ns₂))
    
        elseif length(parameter) == nus₁ * nus₂  # azimuthal resolved declarations
            # Build a matrix for the expanded parameter definition, loop over all unique
            # shell pairs and assign the specified value to the associated interactions.
            mat = Matrix{type}(undef, ns₁, ns₂)
            idxs₁, idxs₂ = enumerate.(_unique_indices.(shells))
            for (o₁,idx₁)=idxs₁, (o₂,idx₂)=idxs₂
                for i=idx₁, j=idx₂
                    mat[i,j] = parameter[o₁, o₂]
                end
            end
            parameter = mat
        
        # One value per shell-pair indicate full declarations; thus no expansion is
        # required. This check acts only as a safeguard against malformed declarations.
        elseif length(parameter) != ns₁ * ns₂
            # Build a, hopefully, helpful error message
            sg, sp = join(size(parameter),"×"), join(species,"-")
            site = species isa Tuple ? "off" : "on"
            error("Malformed $site-site parameter matrix ($sg) given for $sp; expected"*
                  " matrix shapes are $nus₁×$nus₂ (azimuthal) & $ns₁×$ns₂ (full).")
        end

        # Add the full parameter declaration to the new_parameters dictionary
        setindex!(new_parameters, parameter, species)

    end
    return new_parameters
end

end