using MPSKit, MPSKitModels, TensorKit, KrylovKit
using LinearAlgebra, Plots, JLD2
using Logging
global_logger(NullLogger())

include("Functions.jl")

LinearAlgebra.BLAS.set_num_threads(8)

# ─────────────────────────────────────────────
# Parameters (uncomment ARGS block for cluster use)
# ─────────────────────────────────────────────

#=
L           = parse(Int, ARGS[1])
D           = parse(Int, ARGS[2])
n           = parse.(Int, split(ARGS[3], "_"))
calculation = ARGS[4]
=#

L           = 48
D           = 100
n           = [5, 5, 1]
calculation = "DMRG"

t = time()

# ─────────────────────────────────────────────
# Hamiltonian and spectrum
# ─────────────────────────────────────────────

H = periodic_boundary_conditions(heisenberg_hamiltonian(), L)

if calculation == "ED"
    energies, states, sectors = ED_H(H, n)
else
    energies, states, sectors = DMRG_H(H, L, D, n)
end

# ─────────────────────────────────────────────
# Conformal momenta and scaling dimensions
# ─────────────────────────────────────────────

momenta = compute_momenta_mps(energies, states)
S       = momenta ./ (2π / L)

# Velocity: use v = π/2 (exact for Heisenberg), or call estimate_v
v = π / 2

Δ = real.(energies .- energies[1]) ./ (2π * v / L)

# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────

p = plot(;
    legend        = false,
    framestyle    = :box,
    grid          = :y,
    gridalpha     = 0.2,
    dpi           = 300,
)

# Reference lines at integer and half-integer scaling dimensions
max_Δ = ceil(Int, maximum(Δ))
hline!(p, collect(0:max_Δ);        color = "lightgrey", label = "")
hline!(p, collect(0.5:1:max_Δ-0.5); color = "lightgrey", label = "")

# Color and label per SU(2) sector
sector_colors = [:maroon, :firebrick1, :navajowhite, :indianred]
sector_labels = ["Singlets (S=0)", "Triplets (S=1)", "Quintets (S=2)", "Septets (S=3)"]

for s_idx in 0:(length(n) - 1)
    mask = (sectors .== s_idx)
    col  = s_idx + 1 <= length(sector_colors) ? sector_colors[s_idx + 1] : :black
    lab  = s_idx + 1 <= length(sector_labels) ? sector_labels[s_idx + 1] : "S=$s_idx"

    scatter!(p, S[mask], Δ[mask]; label = lab, color = col, markershape = :circle,
             markerstrokewidth = 0.5)

    # Mirror boundary points at ±L/2 (Brillouin zone edge)
    mask_edge = mask .& isapprox.(abs.(S), L / 2; atol = 1e-2)
    if any(mask_edge)
        scatter!(p, fill(-L/2, sum(mask_edge)), Δ[mask_edge]; label = "", color = col,
                 markershape = :circle, markerstrokewidth = 0.5)
        scatter!(p, fill( L/2, sum(mask_edge)), Δ[mask_edge]; label = "", color = col,
                 markershape = :circle, markerstrokewidth = 0.5)
    end
end

# Highlight stress–energy tensor (S=0, |S|≈2) and current (S=1, |S|≈1) operators
function lowest_n_idx(mask, n)
    idx = findall(mask)
    isempty(idx) && return Int[]
    return idx[sortperm(real.(Δ[idx]))][1:min(n, end)]
end

mask_T = (sectors .== 0) .& isapprox.(abs.(S), 2.0; atol = 1e-4)
mask_J = (sectors .== 1) .& isapprox.(abs.(S), 1.0; atol = 1e-4)

idx_T = lowest_n_idx(mask_T, 2)
idx_J = lowest_n_idx(mask_J, 2)

scatter!(p, S[idx_T], Δ[idx_T]; label = "Stress–energy tensor", color = :maroon,
         markershape = :diamond, markersize = 7)
scatter!(p, S[idx_J], Δ[idx_J]; label = "Current", color = :firebrick1,
         markershape = :diamond, markersize = 7)

# Axes and labels
marge = L / 20
stap  = L / 4
plot!(p;
    xlims   = (-(L/2) - marge, (L/2) + marge),
    xticks  = -(L/2):stap:(L/2),
    legend  = :bottomleft,
    legendfontsize      = 8,
    legendtitlefontsize = 9,
)
xlabel!(p, "Momentum S [units of 2π/L]")
ylabel!(p, "Scaling Dimension Δ")
title!(p, "Conformal Spectrum – Heisenberg Chain"; titlefontsize = 12)

# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────

savefig(p, "plot_L$(L)_D$(D)_$(calculation).png")
jldsave("data_L$(L)_D$(D)_$(calculation).jld2"; energies, Δ, S, L, sectors)

@info "Elapsed: $(round(time() - t; digits=1)) s"
p
