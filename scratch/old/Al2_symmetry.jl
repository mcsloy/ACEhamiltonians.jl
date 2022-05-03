using Serialization, Statistics, LinearAlgebra, JuLIP, NeighbourLists, StaticArrays
using ACEhamiltonians.DataIO: load_old_hamiltonian, load_old_atoms
using ACEhamiltonians.DataManipulation: collect_data, collect_data_r, BondState, get_states
using ACEhamiltonians.Bases: envelope
using JuLIP: Atoms
using ACEhamiltonians.Fitting2: fit!, predict


random_flip(v::SVector{3, Float64}) = v .* SVector{3, Float64}(rand([1.0, -1.0], 3))
random_rattle(v::SVector{3, Float64}) = v .+ SVector{3, Float64}(rand(3) .- 0.5)
random_push(v::SVector{3, Float64}) = v .+ SVector{3, Float64}((v ./ norm(v)) * (rand(1)[1] - 0.5))
ops = Function[random_flip, random_rattle, random_push]

random_perturbation(v::SVector{3, Float64}) = rand(ops, 1)[1](v)

# function perturb(basis, states, reference, old_error::Union{Float64, Nothing}=nothing)
#     if isnothing(old_error)
#         old_error = mean(abs.(predict(basis, states) - reference))
#     end 
#     new_states = [states[1], BondState(random_perturbation(states[2].rr), states[2].rr0, :env)]
#     new_error = mean(abs.(predict(basis, new_states) - reference))
#     println(old_error, ", ", new_error)
#     if new_error < old_error
#         return new_states, true, new_error
#     else
#         return states, false, old_error
#     end
# end

function perturb(basis, states, reference, old_error::Union{Float64, Nothing}=nothing)
    if isnothing(old_error)
        old_error = mean(abs.(abs.(predict(basis, states)) - abs.(reference)))
    end 
    new_vector = random_perturbation(states[1].rr)
    new_states = [BondState(new_vector, states[1].rr0, :bond), states[2]]
    new_error = mean(abs.(abs.(predict(basis, new_states)) - abs.(reference)))
    println(old_error, ", ", new_error)
    if new_error < old_error
        println(new_vector .- states[1].rr)
        return new_states, true, new_error
    else
        return states, false, old_error
    end
end



function check_block(basis, s₁, s₂, i, j)
    u = predict(basis, s₁)
    l = predict(basis, s₂)
    ur = H[a2b(i), a2b(j)][b2s(6), b2s(6)]
    lr = H[a2b(j), a2b(i)][b2s(6), b2s(6)]
    e1 = mean(abs.(u - ur))
    e2 = mean(abs.(l - lr))
    e3 = mean(abs.(u - l'))
    e4 = mean(abs.(u - l))
    return (e1, e2, e3, e4)
end



path = "/home/ajmhpc/Projects/ACEtb/Data/new_test_data/build/Al3_7.h5"

_shells = [0, 0, 0, 1, 1, 2]
_norbs = _shells * 2 .+ 1
_ends = cumsum(_norbs)
_starts = _ends - _norbs .+ 1
b2s(i) = _starts[i]:_ends[i]
a2b(i) = (1:14) .+ ((i - 1) * 14)


println("Collecting data")
H, atoms = load_old_hamiltonian(path), load_old_atoms(path)
# atoms = Atoms(;Z=atoms.Z, X=[i .+ 10 for i in atoms.X], cell=atoms.cell)
basis_def = Dict(13=>[0, 0, 0, 1, 1, 2])

basis = deserialize("basis_dd_new.bin")
env = envelope(basis)

ds_12 = collect_data(H, basis, atoms, basis_def, [1, 2]);
ds_21 = collect_data_r(H, basis, atoms, basis_def, [1, 2]);
fit!(basis, ds_12)

# ds_12_2 = collect_data(H, basis_2, atoms, basis_def, [1, 2]);
# ds_21_2 = collect_data_r(H, basis_2, atoms, basis_def, [1, 2]);
# fit!(basis_2, ds_12_2);
# states_a_2 = ds_21_2.states[1];

states_a = ds_21.states[1]

# mean(abs.(predict(basis, states_a) - H[a2b(2), a2b(1)][b2s(6), b2s(6)]))

# gld(i, b) = mean(abs.(predict(b, i) - H[a2b(2), a2b(1)][b2s(6), b2s(6)]))
# gld(i) = mean(abs.(predict(basis, i) - H[a2b(2), a2b(1)][b2s(6), b2s(6)]))

# ns(v) = [states_a[1], BondState(SVector{3, Float64}(v), states_a[2].rr0, :env)]

ns(v, s) = [s[1], BondState(SVector{3, Float64}(v), s[2].rr0, :env)]

gld_2(i) = mean(abs.(predict(basis_2, i) - H[a2b(2), a2b(1)][b2s(6), b2s(6)]))


# states_b = [states_a[1], BondState(SVector{3, Float64}([-0.2, 1.0, 0.0]), states_a[2].rr0, :env)];

# f1(i) = gld(ns([i[1], 1.0, 0.0]))
# f2(i) = gld(ns([i[1], i[2], 0.0]))
# f3(i) = gld(ns([i[1], i[2], i[3]]))

# f4(i) = gld(ns([-1.41702, 0.0, i[1]]))

# states_x = nothing
# r = H[a2b(2), a2b(1)][b2s(6), b2s(6)]

# states_b, improved, e = perturb(basis, states_a, r);
# for _=1:1000
#     global states_b, improved, e = perturb(basis, states_a, r, e);
#     if improved
#         global states_x = states_b
#         println("Improved")
#         println(states_b)
#         break
#     end
# end

# new_state(state, new_v) = [state[1], BondState(new_v, state[2].rr0, :env)]

# push_by(v::SVector{3, Float64}, n::Float64) = v .+ SVector{3, Float64}((v ./ norm(v)) * n)


# Non-offset state errors
#   i)   rr ≡ rr0: 0.016
#   ii)  rr = -(rr0 / 2): 0.014
#   iii) rr = rr0 / 2: 0.014

# function get_states(i::I, j::I, atoms::Atoms, envelope::BondEnvelope,
#     image::Union{AbstractArray{I}, Nothing}=nothing; r::F=20.0) where {I<:Integer, F<:AbstractFloat}
#     # Todo:
#     #   - If the distance between atoms i and j is greater than the cutoff distance r
#     #     then it is likely that an error will be encountered. A safety catch should be
#     #     built in to handle this scenario.
    
#     # Single system no translation vectors
#     pair_list = JuLIP.neighbourlist(atoms, r) # ← TODO: REMOVE
    
#     # Get the bond vector between atoms i & j; where i is in the origin cell & j resides
#     # in either i) closest periodic image, or ii) that specified by `image` if provided.
#     rr0 = atoms.X[j] - atoms.X[i]
#     if isnothing(image)
#         rr0 = project_min(atoms, rr0)
#     else
#         rr0 += (adjoint(image .* atoms.pbc) * atoms.cell).parent
#     end
    
#     # Note this is temporary code that will be removed once bond state inversion
#     # issues have been resolved.
#     offset = nothing
#     rr = nothing
#     if envelope.λ == 0.0
#         offset = rr0 / 2
#         rr = -offset  # Check that this is actually valid
#     else
#         offset = typeof(rr0)(zeros(3))
#         rr = rr0
#         rr = (rr0 / 2)
#     end

#     # Bond state; "rr"=position of i relative j, "rr0"=bond vector & "be" indicates
#     # that this is a bond (rr==rr0 here).
#     bond_state = BondState(rr, rr0, :bond)

#     # Get the indices & distance vectors of atoms neighbouring i
#     idxs, vecs = NeighbourLists.neigs(pair_list, i)

#     # Locate j; done to avoid accidentally including it as an environmental atom
#     j_idx = findfirst(x -> norm(x - rr0) < 1E-10, vecs)

#     # Environmental atoms states, "rr0"=bond=vector, "rr"=atom's position relative to i &
#     # "be" indicates that this is an environmental state.
#     env_states = BondState{typeof(rr0)}[BondState(v - offset, rr0, :env) for v in vecs[1:end .!= j_idx]]
    
#     # Cull states outside of the bond envelope. A double filter is required as the
#     # inbuilt filter operation deviates from standard julia behaviour.
#     env_states = Base.filter(x -> filter(envelope, x), env_states)

#     return [bond_state; env_states]

# end