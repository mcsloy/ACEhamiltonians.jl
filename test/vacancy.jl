include("bands.jl")
using CairoMakie
using ASE, JuLIP
CairoMakie.activate!(type = "svg")

##

cd(joinpath(@__DIR__,"../")) # paths of input files are relative to script location


atoms = Atoms(ASE.read_xyz("fcc-9x9x9-relaxed-geometry.in"))
N_atoms = length(atoms)

n = neighbourlist(atoms, 4.0)
undercoord = findall([sum(n.i .== i) for i=1:length(atoms)] .== 11)

##

using DelimitedFiles

fcc_dos_aims = readdlm("fcc-dos", skipstart=3)
vac_dos_aims = readdlm("vac-dos", skipstart=3)

N_orb = 14
N_eig = N_atoms * N_orb

H_vac, S_vac = h5open("vacancy-relax-fcc.h5") do h5file
    read(h5file, "H"), read(h5file, "S")
end

Hb, Sb = full_to_block(H_vac, S_vac, N_orb)
for bi=1:N_atoms
    Sb[Block(bi, bi)] = I(N_orb)
end

H_vac = Hermitian(Matrix(Hb))
S_vac = Hermitian(Matrix(Sb))
inv_sqrt_S_vac = inv(sqrt(S_vac))
Ht_vac = Hermitian(inv_sqrt_S_vac * H_vac * inv_sqrt_S_vac)
ϵ_vac, ϕ_vac = eigen(Ht_vac)

##

H_p, S_p = h5open("fcc_999_vacancy_compressed.h5") do h5file
    read(h5file, "H"), read(h5file, "S")
end

H_p .+= H_p'
H_p[diagind(H_p)] ./= 2
S_p .+= S_p'
S_p[diagind(S_p)] ./= 2

Hb_pred, Sb_pred = full_to_block(H_p, S_p, N_orb)

for bi=1:N_atoms
    Sb_pred[Block(bi, bi)] = I(N_orb)
end

H_pred = Hermitian(Matrix(Hb_pred))
S_pred = Hermitian(Matrix(Sb_pred))
inv_sqrt_S_pred = inv(sqrt(S_pred))
Ht_pred = Hermitian(inv_sqrt_S_pred * H_pred * inv_sqrt_S_pred)
ϵ_pred, ϕ_pred = eigen(Ht_pred)
##

fig = Figure(resolution=(800,600))

i, j = 1, 2

ax, hm1 = heatmap(fig[1,1], (Hb[Block(i,j)]) )
hidedecorations!(ax)
ax.title = "DFT H / Ha"
ax.aspect = 1
ylims!(N_orb, 0.5)
Colorbar(fig[1,2], hm1)

ax, hm2 = heatmap(fig[1,3], (Hb_pred[Block(i,j)]))
hidedecorations!(ax)
ax.title = "ACE H / Ha"
ax.aspect = 1
ylims!(N_orb, 0.5)
Colorbar(fig[1,4], hm2)

ax, hm3 = heatmap(fig[1,5], log10.(abs.(Hb[Block(i,j)] - Hb_pred[Block(i,j)])))
hidedecorations!(ax)
ax.title = "log10(error / Ha)"
ax.aspect = 1
ylims!(N_orb, 0.5)
Colorbar(fig[1,6], hm3)

ax, hm4 = heatmap(fig[2,1], (Sb[Block(i,j)]))
hidedecorations!(ax)
ax.title = "DFT S"
ax.aspect = 1
ylims!(N_orb, 0.5)
Colorbar(fig[2,2], hm4)

ax, hm5 = heatmap(fig[2,3], (Sb_pred[Block(i,j)]))
hidedecorations!(ax)
ax.title = "ACE S"
ax.aspect = 1
ylims!(N_orb, 0.5)
Colorbar(fig[2,4], hm5)

if i != j
    ax, hm6 = heatmap(fig[2,5], log10.(abs.(Sb[Block(i,j)] - Sb_pred[Block(i,j)])))
    hidedecorations!(ax)
    ax.title = "log10(error)"
    ax.aspect = 1
    ylims!(N_orb, 0.5)
    Colorbar(fig[2,6], hm6)
end

resize_to_layout!(fig)
fig
##

errs = zeros(N_atoms, N_atoms)
for j=1:N_atoms, i=1:N_atoms
    errs[j, i] = maximum(abs.(Hb[Block(i, j)] - Hb_pred[Block(i, j)]))
end

fig = Figure()
ax, hm = heatmap(fig[1,1], log10.(errs))
#hidedecorations!(ax)
ax.title = "log10(MAE block_ij / Ha)"
ax.aspect = 1
ylims!(N_atoms, 0.5)
Colorbar(fig[1,2], hm)
fig

##

dϵ_vac = eigvals_error(ϵ_pred, H_pred, H_vac, S_pred, S_vac)

fig = Figure()
ax = Axis(fig[1,1])
l1 = lines!(ax, 1..N_eig, ϵ_vac, label="DFT spectrum")
l2 = lines!(ax, 1..N_eig, ϵ_pred, label="ACE spectrum")
fill_between!(ax, 1:N_eig, ϵ_pred, ϵ_pred + dϵ_vac, color=l2.attributes.color)
axislegend(position=:lt)
ax.xlabel = "Eigenvalue index"
ax.ylabel = "Eigenvalue / Ha"
ylims!(ax, -20, 20)
fig
##

σ = 0.01
egrid = range(-1,1,length=2001)

δ(ϵ) = exp.(-((egrid .- ϵ) ./ σ).^2) ./ (sqrt(π) * σ)

fulldos(ϵ) = sum([δ(ϵ[i]) for i=1:length(ϵ)])

function pdos(ϵ, ϕ, atoms)
    dos = zeros(length(egrid))
    for i=1:length(ϵ)
        for a in atoms # list of 1, 8, 9, 17, 72, 73, 81, 89, 153, 649, 657
            for j = 1:N_orb
                dos .+= abs(ϕ[(a-1) * N_orb + j, i])^2 * δ(ϵ[i])
            end
        end
    end
    dos
end

##

H_fcc, S_fcc = h5open("FCC-supercell-000.h5") do h5file
    dropdims(read(h5file, "aitb/H"), dims=3), dropdims(read(h5file, "aitb/S"), dims=3)
end

H_fcc = Hermitian(H_fcc)
S_fcc = Hermitian(S_fcc)
inv_sqrt_S_fcc = inv(sqrt(S_fcc))
Ht_fcc = Hermitian(inv_sqrt_S_fcc * H_fcc * inv_sqrt_S_fcc)
ϵ_fcc, ϕ_fcc = eigen(Ht_fcc)


##

ϵf_fcc = fermi_level(ϵ_fcc, [1], 13 * 729, 0.01)
ϵf_vac = fermi_level(ϵ_vac, [1], 13 * 728, 0.01)
ϵf_pred = fermi_level(ϵ_pred, [1], 13 * 728, 0.01)

##

fig = Figure()
ax1 = Axis(fig[1, 1])

scale =  length(atoms) / length(undercoord)

lines!(ax1, (egrid .- ϵf_fcc).* hartree2ev, fulldos(ϵ_fcc), label="DFT Full DOS FCC", linewidth=2)
lines!(ax1, (egrid .- ϵf_vac).* hartree2ev, fulldos(ϵ_vac), label="DFT Full DOS vacancy", linewidth=2)
lines!(ax1, (egrid .- ϵf_vac).* hartree2ev, scale * pdos(ϵ_vac, ϕ_vac, undercoord), label="DFT PDOS vacancy neighbours", linewidth=2)

lines!(ax1, (egrid .- ϵf_vac).* hartree2ev, fulldos(ϵ_pred), label="ACE Full DOS vacancy", linewidth=2)
lines!(ax1, (egrid .- ϵf_vac).* hartree2ev, scale * pdos(ϵ_pred, ϕ_pred, undercoord), label="ACE PDOS vacancy", linewidth=2)

axislegend(ax1, ; position=:lt)
xlims!(ax1, -20, 30)

ax1.xlabel = "(ϵ - ϵf) / eV"
fig
xlims!(ax1, -20, 0)
fig
