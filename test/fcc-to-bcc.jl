include("bands.jl")
using CairoMakie
CairoMakie.activate!(type = "svg")

##

cd(@__DIR__) # paths of input files are relative to script location

N_orb = 14
kmesh = [9, 9, 9]

H, S = h5open("BCC-supercell-000.h5") do h5file
    dropdims(read(h5file, "aitb/H"), dims=3), dropdims(read(h5file, "aitb/S"), dims=3)
end
Hb, Sb = full_to_block(H, S, N_orb)
R = tb_cells(kmesh)
H_NMM_bcc, S_NMM_bcc, blocks = split_HS(Hb, Sb, R)  # BCC reference HS

# set onsite S to identity - remove numerical noise
S_NMM_bcc[365, :, :] = I(14)

H, S = h5open("FCC-supercell-000.h5") do h5file
    dropdims(read(h5file, "aitb/H"), dims=3), dropdims(read(h5file, "aitb/S"), dims=3)
end
Hb, Sb = full_to_block(H, S, N_orb)
R = tb_cells(kmesh)
H_NMM_fcc, S_NMM_fcc, blocks = split_HS(Hb, Sb, R) # FCC reference HS

# set onsite S to identity - remove numerical noise
S_NMM_fcc[365, :, :] = I(14)

fcc_kpath = KPath(:FCC)
bcc_kpath = KPath(:BCC)

## 

# density of states and Fermi level
# BCC
ϵ_kM_bcc, w_k, ϵgrid, d_bcc_fine = density_of_states(kmesh, H_NMM_bcc, S_NMM_bcc, R, 
                                        egrid=range(-1,1,length=1000), σ=0.02)

ϵf_bcc = fermi_level(ϵ_kM_bcc, w_k, 13, 0.02)

# FCC
ϵ_kM_fcc, w_k_fcc, ϵgrid_fcc, d_fcc_fine = density_of_states(kmesh, H_NMM_fcc, S_NMM_fcc, R, 
                                        egrid=range(-1,1,length=1000), σ=0.02)

ϵf_fcc = fermi_level(ϵ_kM_fcc, w_k_fcc, 13, 0.02)


# Exact DFT band structure

# BCC
bands_bcc = bandstructure(bcc_kpath, H_NMM_bcc, S_NMM_bcc, R)

# FCC
bands_fcc = bandstructure(fcc_kpath, H_NMM_fcc, S_NMM_fcc, R)

##

dos = []
efermi = []
allbands = []
allbands_err = []
eigs = []
E_band = []

dos_ref = []
efermi_ref = []
allbands_ref = []
eigs_ref = []
E_band_ref = []

for i = 0:15
    println("Reading config $i")
    idx = @sprintf "%03d" i
    filename = "ACEtb-BCC-to-FCC/out_$(idx)_2_9.h5"
    H, S = h5open(filename) do h5file
        real.(permutedims(read(h5file, "H"), (3, 1, 2))), real.(permutedims(read(h5file, "S"), (3, 1, 2)))
    end
    S[365, :, :] = I(14)
    ϵp_kM, w_k, ϵgrid, d_p = density_of_states(kmesh, H, S, R,
                                               egrid=range(-1,1,length=1000), σ=0.02)
    ϵf_p = fermi_level(ϵp_kM, w_k, 13, 0.01)

    println("Reading ref config $i")
    ref_filename = "/home/eng/essmpc/AITB/FHI-aims/Al/BCC-to-FCC-transition/CORRECTED/$(idx)/SK-supercell-$(idx).h5"
    H_ref, S_ref = h5open(ref_filename) do h5file
        dropdims(read(h5file, "aitb/H"), dims=3), dropdims(read(h5file, "aitb/S"), dims=3)
    end
    # H_ref, S_ref = read_FHIaims_HS_out(joinpath("FHIaims-BCC-to-FCC-for-Reini/", idx))
    Hb, Sb = full_to_block(H_ref, S_ref, N_orb)
    H_ref, S_ref, blocks = split_HS(Hb, Sb, R)
    S_ref[365, :, :] = I(14)

    ϵ_kM_ref, w_k, ϵgrid, d = density_of_states(kmesh, H_ref, S_ref, R,
                                                   egrid=range(-1,1,length=1000), σ=0.02)
    ϵf_ref = fermi_level(ϵp_kM, w_k, 13, 0.01)


    push!(eigs, ϵp_kM)
    push!(E_band, sum(fermi.(ϵp_kM, ϵf_p, 0.01) .* ϵp_kM .* w_k))
    push!(dos, d_p)
    push!(efermi, ϵf_p)
    bands, bands_err = bandstructure(bcc_kpath, H, S, R, H0_NMM=H_ref, S0_NMM=S_ref)
    push!(allbands, bands)
    push!(allbands_err, bands_err)

    push!(eigs_ref, ϵ_kM_ref)
    push!(E_band_ref, sum(fermi.(ϵ_kM_ref, ϵf_ref, 0.01) .* ϵ_kM_ref .* w_k))
    push!(dos_ref, d)
    push!(efermi_ref, ϵf_ref)
    bands_ref = bandstructure(bcc_kpath, H_ref, S_ref, R)
    push!(allbands_ref, bands_ref)
end

fcc_atoms = load_json("FCC-999-supercell.json")
bcc_atoms = load_json("BCC-999-supercell.json")

atoms = [ load_json("ACEtb-BCC-to-FCC/geometry-$(@sprintf "%03d" i).json") for i=0:15 ]

cell(i) = hcat(hcat(atoms[i]["cell"])...)

function c_over_a(i) 
    L = cell(i)
    a = norm(L[:, 3])
    c = norm(L[:, 1])
    return c/a
end

ca = c_over_a.(1:16)

##

function read_matrix(filename; dim=729*14)
    M = zeros(dim, dim)
    open(filename) do file
        for line in eachline(file)
            i, j, v = split(line)
            i, j, v = parse(Int, i), parse(Int, j), parse(Float64, v)
            M[i, j] = v
        end
    end
    return M
end

function read_FHIaims_HS_out(dirname; dim=729*14)
    H_filename = joinpath(dirname, "hamiltonian.out")
    S_filename = joinpath(dirname, "overlap-matrix.out")
    H = read_matrix(H_filename; dim=dim)
    S = read_matrix(S_filename; dim=dim)
    return H, S
end

##

using FileIO

egrid = range(-1,1,length=1000)

function plot_fcc_bcc!(ax, i; color=:black, ylabel="")
    ax.ylabel = ylabel
    lines!(ax, (egrid .- efermi_ref[i+1]) .* hartree2ev, dos_ref[i+1], color=:red, label="DFT")
    lines!(ax, (egrid .- efermi[i+1]) .* hartree2ev, dos[i+1], color=:blue)
    # vline!([efermi[i+1]] * hartree2ev, color=:red, ls=:dash, label=nothing)
    vlines!(ax, [0.0] * hartree2ev, color=color, linestyle=:dash, linewidth=2, label=nothing)
    ax.xticks = [-10, 0, 10, 20]
    xlims!(ax, -15, 30)
    ylims!(ax, 0, 15)
end

function image_fcc_bcc!(ax, i)
    idx = @sprintf "%04d" i
    img = FileIO.load("bain-path$idx.png")
    ax.aspect = DataAspect()
    ax.yreversed = true
    image!(ax, img')
    hidedecorations!(ax)
    # hidespines!(ax)
end

fig = Figure(resolution=(200, 300))

ax1 = Axis(fig[1,1], title="c/a = " * @sprintf("%.2f",c_over_a(i+1)))
ax2 = Axis(fig[2,1])

plot_fcc_bcc!(ax1, 0)
image_fcc_bcc!(ax2, 0)

rowsize!(fig.layout, 2, Auto(2))

fig
##

using PyCall
@pyimport scipy.stats as st


function band_error(bands, bands_ref, ϵf, ϵf_ref; σ=0.003166790852369942, range=2:14)  # equivalent to 1000 K
    norm([norm((bands[:, i] .* fermi.(bands[:, i], ϵf, σ)) - 
               (bands_ref[:, i] .* fermi.(bands_ref[:, i], ϵf_ref, σ))) for i ∈ range]) ./ sqrt(length(bands))
end

dos_error(dos, dos_ref) = st.wasserstein_distance(dos, dos_ref)

dos_err = [dos_error(dos[i], dos_ref[i]) for i= 1:16]

band_err = [ hartree2ev * band_error(allbands[i], allbands_ref[i], efermi[i], efermi_ref[i]) for i=1:16]

##

# dos_diff = plot(ca, [dot(d[egrid .< ϵf_fcc], d_fcc_fine[egrid .< ϵf_fcc])/norm(d[egrid .< ϵf_fcc])/norm(d_fcc_fine[egrid .< ϵf_fcc]) for d in dos], 
#         label=raw"$D \cdot D_\mathrm{FCC}$", color=3, lw=2, xlabel=raw"$c/a$", ylabel="DoS overlap")
# plot!(ca, [dot(d[egrid .< ϵf_fcc], d_bcc_fine[egrid .< ϵf_fcc])/norm(d[egrid .< ϵf_fcc])/norm(d_bcc_fine[egrid .< ϵf_fcc]) for d in dos], 
#         label=raw"$D \cdot D_\mathrm{BCC}$", color=4, lw=2)

noto_sans = assetpath("fonts", "NotoSans-Regular.ttf")
noto_sans_bold = assetpath("fonts", "NotoSans-Bold.ttf")

fig = Figure(resolution=(600, 700), font=noto_sans)
for (i, (err, label)) in enumerate(zip([dos_err, band_err], ["DOS error", "Band error / eV"]))
    ax = Axis(fig[i+1, 1], xlabel = i == 1 ? "" : "Bain path reaction coordinate c/a", ylabel=label, grid=false)
    lines!(ax, ca, err, label="Error in ACE model with respect to DFT", color=:blue, linewidth=2)
    vlines!(ax, [1.0], color=:green, linestyle=:dash, linewidth=2, label="FCC")
    vlines!(ax, [1.0/sqrt(2)], color=:purple, linestyle=:dash, linewidth=2, label="BCC")
    vlines!(ax, [ca[5+1], ca[9+1]], color=:black, linestyle=:dash, linewidth=2)
    xlims!(ax, 0.66, 1.05)
    ax.xticks =  [1/sqrt(2), ca[5+1], ca[9+1], 1.0]
    ax.xtickformat = xs -> [@sprintf "%.2f" x for x in xs]
    hidexdecorations!(ax, grid=true, label=false, ticklabels=i == 1, ticks=false)
    hideydecorations!(ax, grid=true, label=false, ticklabels=false, ticks=false)
end
fig[1, 1] = Legend(fig, fig.content[1], framevisible=false, orientation=:horizontal)

f = fig[4, 1] = GridLayout()
# g = fig[5, 1]

ha = [-0.08, 0.25, 0.73, 1.05]
va = [0.82, 0.5, 0.5, 0.82]

for (j, i) in enumerate([2, 5, 9, 12])
    color = :black
    i == 2 && (color = :purple)
    i == 12 && (color = :green)
    ax = Axis(f[1, j])
    plot_fcc_bcc!(ax, i; color=color, ylabel=j == 1 ? "DoS" : "")
    hidexdecorations!(ax, grid=true, label=false, ticklabels=false, ticks=false)
    hideydecorations!(ax, grid=true, label=false, ticklabels=false, ticks=false)

    img_ax = Axis(fig[2, 1], width=Relative(0.35), height=Relative(0.35), halign=ha[j], valign=va[j])
    image_fcc_bcc!(img_ax, i)
end

for (label, layout) in zip(["(a)", "(b)", "(c)"], [fig[2,1], fig[3,1], fig[4,1]])
    Label(layout[1, 1, TopLeft()], label,
        textsize = 18,
        # font = noto_sans_bold,
        padding = (0, 5, 10, 0),
        halign = :right)
end

Label(fig[4, 1, Bottom()], "Energy relative to Fermi energy / eV", halign=:center, valign=:top, padding=(0, 0, 0, 30))

rowsize!(fig.layout, 1, 20)
rowsize!(fig.layout, 4, Auto(0.8))
rowgap!(fig.layout, 1, 0)
rowgap!(fig.layout, 0)
colgap!(f, 10)

save("BCC_to_FCC.pdf", fig)
fig
##

band_energy(ϵ_kM, ϵf, σ) = sum(fermi.(ϵ_kM, ϵf, σ) .* ϵ_kM .* w_k)

E_band = [band_energy(eig, ϵf, 0.01) for (eig, ϵf) in zip(eigs, efermi)]
E_band_ref =  [band_energy(eig, ϵf, 0.01) for (eig, ϵf) in zip(eigs_ref, efermi_ref)]

E0 = minimum(E_band_ref)

function plot_eband(i=nothing)
    plot(ca, hartree2ev * (E_band_ref .- minimum(E_band_ref)) ./ 729 * 1e3, color=:red, label="DFT",
            xlabel=raw"$c/a$", ylabel="Band energy / meV/atom", legend=:topright)
    plot!(ca, hartree2ev * (E_band .- minimum(E_band)) ./ 729 * 1e3, color=:blue,label="ACE")
    # ylims!(-1, 5)
    vline!([ca[2+1]], color=3, ls=:dash, label="FCC")
    vline!([ca[12+1]], color=4, ls=:dash, label="BCC")
    vline!([ca[5+1], ca[9+1] ], color=:black, ls=:dash, label=nothing)

    # if i !== nothing
    #     scatter!([ca[i]], hartree2ev * (E_band_ref[i] .- E0) ./ 729 * 1e3, marker=:o, color=:red)
    #     scatter!([ca[i]], hartree2ev * (E_band[i] .- E0) ./ 729 * 1e3, marker=:o, color=:blue)
    # end
end

band_error_plt = plot_eband()
savefig("band_energy.pdf")
band_error_plt
##

##


##

function compare_bands(i)
    plt = plot_bands(bcc_kpath, allbands[i+1] * hartree2ev, color=:blue, fermi_level=efermi[i+1])
    plot_bands!(plt, bcc_kpath, allbands_ref[i+1] * hartree2ev, color=:red, fermi_level=efermi_ref[i+1])
    ylims!(plt, -20, 25)
    return plt
end

function compare_dos(i)
    plt = plot(dos_ref[i+1], (egrid .- efermi_ref[i+1]) .* hartree2ev, color=:red, label="DFT")
    plot!(dos[i+1], (egrid .- efermi[i+1]) .* hartree2ev, color=:blue, label="ACE") 
    title!(raw"$c/a = " * @sprintf("%.2f",c_over_a(i+1)) * raw"$", font=font(12))
    return plt
end

l = @layout[a{0.7w,1h}  b{0.3w,1h}]

fcc_bcc_dos = @animate for i in 0:15
    p1 = compare_bands(i)
    p2 = compare_dos(i)
    plot(p1, p2, layout=l)
end

gif(fcc_bcc_dos, "FCC_BCC_DOS.gif", fps=2)

##
