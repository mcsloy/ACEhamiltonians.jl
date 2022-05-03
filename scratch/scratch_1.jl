using Serialization, Statistics, LinearAlgebra, JuLIP, NeighbourLists, ACE, ACEhamiltonians
using JuLIP.Utils: project_min
import ACE: evaluate!
import ACEbase: evaluate!
using ACEhamiltonians.Fitting: evaluateval_real, solve_ls
using ACEhamiltonians.Bases: Basis, off_site_ace_basis, envelope
using ACEhamiltonians.DataIO: load_old_hamiltonian, load_old_atoms
using ACEhamiltonians.DataManipulation: collect_data, collect_data_r, get_states, DataSet, BondState

using ACE: CylindricalBondEnvelope, Rn1pBasis, Ylm1pBasis, ACEConfig, evaluate, scaling, AbstractState

using ACEhamiltonians.Fitting2: fit!, predict, predict!
import ACEhamiltonians.Fitting2: predict

function evaluateval_real_2(Aval)

    n₁, n₂ = size(Aval[1])
    ℓ₁, ℓ₂ = Int((n₁ - 1) / 2), Int((n₂ - 1) / 2)

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

        Aval_real[k] = (Aval_real[k] + Aval_real[k]') / 2

        # Prefactor "scale" could be applied here as a matrix operation
    end

    if norm(Aval_real - real(Aval_real))<1e-12
        return real(Aval_real)
    else
        error("norm = $(norm(Aval_real - real(Aval_real))), please recheck...")
    end
end


function assemble_ls_2(basis::Basis, data::DataSet, zero_mean::Bool=false)
    # This will be rewritten once the other code has been refactored.

    # Should `A` not be constructed using `acquire_B!`?

    n₀, n₁, n₂ = size(data)
    # Currently the code desires "A" to be an X×Y matrix of Nᵢ×Nⱼ matrices, where X is
    # the number of sub-block samples, Y is equal to `size(bos.basis.A2Bmap)[1]`, and
    # Nᵢ×Nⱼ is the sub-block shape; i.e. 3×3 for pp interactions. This may be refactored
    # at a later data if this layout is not found to be strictly necessary.
    cfg = ACEConfig.(data.states)
    Aval = evaluate.(Ref(basis.basis), cfg)
    A = permutedims(reduce(hcat, evaluateval_real_2.(Aval)), (2, 1))
    # A = permutedims(reduce(hcat, evaluateval_real.(Aval)), (2, 1))
    
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


function fit_2!(basis::Basis, data::DataSet, zero_mean::Bool=false)
    # Lambda term should not be hardcoded to 1e-7!

    # Get the basis function's scaling factor (?)
    Γ = Diagonal(scaling(basis.basis, 2))

    # Setup the least squares problem
    Φ, Y, x̄ = assemble_ls_2(basis, data, zero_mean)
    
    # Assign the mean value to the basis set
    basis.mean = x̄

    # Solve the least squares problem and get the coefficients
    basis.coefficients = collect(solve_ls(Φ, Y, 1e-7, Γ, "LSQR"))

    nothing
end

function predict_2!(values::Matrix{F}, basis::Basis, states::Vector{S}) where {F<:AbstractFloat, S<:AbstractState}
    A = evaluate(basis.basis, ACEConfig(states))
    B = evaluateval_real_2(A)
    values .= (basis.coefficients' * B) + basis.mean
end


# Construct and fill a matrix with the results of a single state
function predict_2(basis::Basis, states::Vector{S}) where S<:AbstractState
    n, m, type = ACE.valtype(basis.basis).parameters[3:5]
    values = Matrix{real(type)}(undef, n, m)
    predict_2!(values, basis, states)
    return values
end


function check_block(basis, s₁, s₂, i, j)
    u = predict(basis, s₁)
    l = predict(basis, s₂)
    ur = H[a2b(i), a2b(j)][b2s(5), b2s(5)]
    lr = H[a2b(j), a2b(i)][b2s(5), b2s(5)]
    e1 = mean(abs.(u - ur))
    e2 = mean(abs.(l - lr))
    e3 = mean(abs.(u - l'))
    e4 = mean(abs.(u - l))
    return (e1, e2, e3, e4)
end

function check_block_2(basis, s₁, s₂, i, j)
    u = predict_2(basis, s₁)
    l = predict_2(basis, s₂)
    ur = H[a2b(i), a2b(j)][b2s(5), b2s(5)]
    lr = H[a2b(j), a2b(i)][b2s(5), b2s(5)]
    e1 = mean(abs.(u - ur))
    e2 = mean(abs.(l - lr))
    e3 = mean(abs.(u - l'))
    e4 = mean(abs.(u - l))
    return (e1, e2, e3, e4)
end


function blocks(basis, s₁, s₂)
    return predict(basis, s₁), predict(basis, s₂)
end



path = "/home/ajmhpc/Projects/ACEtb/Data/test_data/FCC-MD-500K/SK-supercell-001.h5"
_shells = [0, 0, 0, 1, 1, 2]
_norbs = _shells * 2 .+ 1
_ends = cumsum(_norbs)
_starts = _ends - _norbs .+ 1
b2s(i) = _starts[i]:_ends[i]
a2b(i) = (1:14) .+ ((i - 1) * 14)

println("Collecting data")
H, atoms = load_old_hamiltonian(path), load_old_atoms(path)
basis_def = Dict(13=>[0, 0, 0, 1, 1, 2])


if isfile("basis_pp.bin")
    println("Loading model")
    basis = deserialize("basis_pp.bin")
    env = envelope(basis)
else
    println("Building model")
    env = CylindricalBondEnvelope(18.0, 12.0, 5.0, λ=0.0, floppy=false)
    basis = Basis(off_site_ace_basis(1, 1, 2, 8, env, 0.1), (13, 13, 5, 5))
    serialize("basis_pp.bin", basis)
end


ds_12 = collect_data(H, basis, atoms, basis_def, [1, 2]);
ds_21 = collect_data_r(H, basis, atoms, basis_def, [1, 2]);

ds_23 = collect_data(H, basis, atoms, basis_def, [2, 3]);
ds_32 = collect_data_r(H, basis, atoms, basis_def, [2, 3]);


ds_12_b = DataSet(ds_12.values, ds_12.block_indices, ds_12.cell_indices, [[ds_21.states[1]; ds_12.states[1]]])
ds_21_b = DataSet(ds_21.values, ds_21.block_indices, ds_21.cell_indices, [[ds_12.states[1]; ds_21.states[1]]])


ds_bu = collect_data(H, basis, atoms, basis_def, [1, 2, 3, 4]);
ds_bl = collect_data_r(H, basis, atoms, basis_def, [1, 2, 3, 4]);
# check_block(basis, ds_12.states[1], ds_21.states[1], 1, 2)

# ds_12_b = DataSet(ds_12.values, ds_12.block_indices, ds_12.cell_indices, [[BondState(ds_12.states[1][1].rr .* sign.(ds_12.states[1][1].rr), ds_12.states[1][1].rr0, :bond) ; ds_12.states[1][2:end]]])
# ds_21_b = DataSet(ds_21.values, ds_21.block_indices, ds_21.cell_indices, [[BondState(ds_21.states[1][1].rr .* sign.(ds_21.states[1][1].rr), ds_21.states[1][1].rr0, :bond) ; ds_21.states[1][2:end]]])

# fit!(basis, ds_12); check_block(basis, ds_12.states[1], ds_21.states[1], 1, 2)
# fit!(basis, ds_12a); check_block(basis, ds_12a.states[1], ds_21a.states[1], 1, 2)
# fit!(basis, ds_12b); check_block(basis, ds_12b.states[1], ds_21b.states[1], 1, 2)

# ds_12_b = DataSet(ds_12.values, ds_12.block_indices, ds_12.cell_indices, [[BondState(ds_12.states[1][1].rr, ds_12.states[1][1].rr0, :bond) ; ds_12.states[1][2:end]]])