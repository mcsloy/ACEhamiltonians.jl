using HDF5, ACEbase, ACEhamiltonians, StaticArrays, Statistics, LinearAlgebra
using HDF5: Group
using ACE: ACEConfig, evaluate, scaling, valtype, State
using ACEhamiltonians.Fitting: evaluateval_real, solve_ls

function evaluateval_real_new(Aval)

    n₁, n₂ = size(Aval[1])

    # allocate Aval_real
    Aval_real = [zeros(ComplexF64, n₁, n₂) for i = 1:length(Aval)]
    # reconstruct real A
    # TODO: I believe that there must exist a matrix form... But I will keep it for now...
    for k=1:length(Aval)

        A = Aval[k].val
        for i=1:n₁, j=1:n₂
            # Magnetic quantum numbers
            m₁, m₂ = i - ℓ₁ - 1, j - ℓ₂ - 1
            
            val = A[i,j]
            
            if m₁ < 0
                val -= (-1)^(m₁) * A[end-i+1,j]
            elseif m₁ > 0
                val += (-1)^(m₁) * A[end-i+1,j]
            end

            if (m₁ > 0 < m₂) || (m₁ < 0 > m₂)
                val += (-1)^(m₁ + m₂) * A[end-i+1, end-j+1]
            elseif (m₁ > 0 > m₂) || (m₁ < 0 < m₂)
                val -= (-1)^(m₁ + m₂) * A[end-i+1, end-j+1]
            end
            if m₂ < 0
                val -= (-1)^(m₂) * A[i, end-j+1]
            elseif m₂ > 0
                val += (-1)^(m₂) * A[i, end-j+1]
            end

            # This could be applied as a matrix to the full block
            s = ((m₁ >= 0) && (m₂ < 0)) ? -1 : 1 # Account for sign flip until mess is sorted
            plane = (m₁ >= 0) ⊻ (m₂ >= 0) ? im : 1
            scale = m₁ == 0 ? (m₂ == 0 ? 1 : 1/√2) : (m₂ == 0 ? 1/√2 : 1/2)

            Aval_real[k][i,j] = scale * plane * (s * (val))
        end
    end
    #return Aval_real
    if norm(Aval_real - real(Aval_real))<1e-12
        return real(Aval_real)
    else
        error("norm = $(norm(Aval_real - real(Aval_real))), please recheck...")
    end
end


# +1 * A[k,j] always

# -1 * s₁*A[α,j] m₁<0
#  0 * s₁*A[α,j] m₁=0
# +1 * s₁*A[α,j] m₁>0

# -1 * s₂*A[i,β] m₂<0
#  0 * s₂*A[i,β] m₂=0
# +1 * s₂*A[i,β] m₂>0

# -1 * s₃*A[α,β] m₁ > 0 > m₂ or m₁ < 0 < m₂
#  0 * s₃*A[α,β] m₁=0 or m₂=0
# +1 * s₃*A[α,β] m₁ > 0 < m₂) or (m₁ < 0 > m₂

# m₁<0:
#   - m₂<0 : 1/2  (A)
#   - m₂=0 : i/√2 (B)
#   - m₂>0 : i/2  (C)

# m₁=0:
#   - m₂<0 : i/√2 (D)
#   - m₂=0 : 1    (E)
#   - m₂>0 : 1/√2 (F)

# m₁>0:
#   - m₂<0 : i/2  (G)
#   - m₂=0 : 1/√2 (H)
#   - m₂>0 : 1/2  (I)

# if kc₁ && jc₁ # A (1/2)
#     val += - s₁*A[α,j] - s₂*A[k,β] + s₃*A[α,β]
# elseif kc₁ && jc₂ # B (im/sqrt(2))
#     val += - s₁*A[α,j]
# elseif kc₁ && jc₃ # C (im/2)
#     val += - s₁*A[α,j] + s₂*A[k,β] - s₃*A[α,β]

# elseif kc₂ && jc₁ # D (im/sqrt(2))
#     val += - s₂*A[k,β]
# # elseif kc₂ && jc₂ # E (1)
# #     val = A[k,j]
# elseif kc₂ && jc₃ # F (1/sqrt(2))
#     val += + s₂*A[k,β]

# elseif kc₃ && jc₁ # G (im/2)
#     val += + s₁*A[α,j] - s₂*A[k,β] - s₃*A[α,β]
# elseif kc₃ && jc₂ # H (1/sqrt(2))
#     val += + s₁*A[α,j]
# elseif kc₃ && jc₃ # I (1/2)
#     val += + s₁*A[α,j] + s₂*A[k,β] + s₃*A[α,β]
    
# end



function assemble_ls(basis::Basis, data::DataSet, zero_mean::Bool=false)
    # This will be rewritten once the other code has been refactored.

    n₀, n₁, n₂ = size(data)
    # Currently the code desires "A" to be an X×Y matrix of Nᵢ×Nⱼ matrices, where X is
    # the number of sub-block samples, Y is equal to `size(bos.basis.A2Bmap)[1]`, and
    # Nᵢ×Nⱼ is the sub-block shape; i.e. 3×3 for pp interactions. This may be refactored
    # at a later data if this layout is not found to be strictly necessary.
    cfg = ACEConfig.(data.states)
    Aval = evaluate.(Ref(basis.basis), cfg)
    A = permutedims(reduce(hcat, evaluateval_real.(Aval)), (2, 1))

    Y = [data.values[i, :, :] for i in 1:n₀]

    # Calculate the mean value x̄
    if !zero_mean && n₁ ≡ n₂ && ison(basis) 
        x̄ = mean(diag(mean(Y)))*I(n₁)
    else
        x̄ = zeros(n₁, n₂)
    end

    Y .-= Ref(x̄)
    return A, Y, x̄
end


function fit!(basis::Basis, data::DataSet, zero_mean::Bool=false)
    # Get the basis function's scaling factor (?)
    Γ = Diagonal(scaling(basis.basis, 2))

    # Setup the least squares problem
    Φ, Y, x̄ = assemble_ls(basis, data, zero_mean)
    
    # Assign the mean value to the basis set
    basis.mean = x̄

    # Solve the least squares problem and get the coefficients
    basis.coefficients = collect(solve_ls(Φ, Y, 1e-7, Γ, "LSQR"))
end


"""
Todo:
    - It will be important to check that the relevant data exists before trying to
      extract it; i.e. don't bother trying to gather carbon on-site data from a H2
      system etc.
    - Currently the basis set definition is loaded for the first system under the
      assumption that it constant across all systems. However, this will break down
      if different species are present in each system, i.e. if the first system is
      H2 but the second is CH4 then the carbon basis set will be absent.
    - The approach taken here limits io overhead by reducing redundant load operations.
      However, this will likely use considerably more memory.
"""
function fit!(model::Model, systems::Vector{Group}, target::Symbol)
    # Section 1: Gather the data

    @assert target in [:H, :S]
    on_site_data = Dict{Basis, DataSet}()
    off_site_data = Dict{Basis, DataSet}()

    add_data!(dict, key, data) = dict[key] = data + getkey(dict, key, nothing) 

    # Load in the basis set (WARNING: currently assumes all systems have at least one of each species)
    basis_def = load_basis_set_definition(systems[1])

    get_matrix = target ≡ :H ? load_hamiltonian : load_overlap

    # Loop over the specified systems
    for system in systems
        # Load the required data from the database entry
        matrix = get_matrix(system)
        atoms = load_atoms(system)
        images = gamma_only(system) ? [1 1 1] : load_cell_translations(system)

        # Loop over the on site bases and collect the appropriate data
        for basis in values(model.on_site_models)
            data_set = collect_data(matrix, basis, atoms, basis_def, images)
            add_data!(on_site_data, basis, data_set)
        end 

        # Repeat for the off-site models
        for basis in values(model.off_site_models)
            data_set = collect_data(matrix, basis, atoms, basis_def, images)
            add_data!(off_site_data, basis, data_set)
        end         
    end

    # Fit the on-site models
    for (basis, data_set) in on_site_data
        fit!(basis, data_set)
    end

    for (basis, data_set) in off_site_data
        fit!(basis, data_set)
    end
end

function predict!(values::Matrix, basis::Basis, states::Vector{Vector{State}})
    for (n, state) in enumerate(ACEConfig.(states))
        A = evaluate(basis.basis, state)
        B = evaluateval_real(A)
        values[n] = (basis.coefficients' * B) + basis.mean
    end
end

function predict(basis::Basis, states::Vector{Vector{State}})
    # Work out the sub-block shape. There must be a better way of doing this. I expect
    # there is a function that can extract the size of the internal sparse matrix. But
    # for now we cheat by pre-evaluating the first state and checking the resulting shape.
    n₁, n₂ = size(evaluateval_real(evaluate(basis.basis, ACEConfig(states[1])))[1])
    values = AbstractArray{valtype(basis.basis), 3}(undef, length(states), n₁, n₂)
    predict!(values, basis, states)
    return values

end

path = "/home/ajmhpc/Documents/Work/Projects/ACEtb/Data/Si/Construction/batch_0.h5"
db = h5open(path, "r")
model = read_dict(load_json("model.json"))
basis = model.on_site_models[(14, 4, 4)]
# basis = read_dict(load_json("basis_on.json"))
# fit!(model, [db["0224"]], :H)

data_set = collect_data_from_hdf5(db["0224"], basis, :H)
fit!(basis, data_set)
# values = predict(basis, data_set.states)

close(db)

