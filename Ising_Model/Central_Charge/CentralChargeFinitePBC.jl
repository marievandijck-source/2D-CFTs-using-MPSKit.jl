using MPSKit, MPSKitModels, TensorKit, Plots
using Logging
global_logger(NullLogger())

# ─────────────────────────────────────────────
# Central charge — critical Ising chain (PBC)
#
# Cardy–Calabrese formula (PBC):
#   S(ℓ) = (c/3) log[(L/π) sin(πℓ/L)] + const
# Exact value:  c = 1/2
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────

L       = 10
Ds      = 6:4:10
fitcuts = L ÷ 3 : 2L ÷ 3

# ─────────────────────────────────────────────
# Hamiltonian
# ─────────────────────────────────────────────

H    = periodic_boundary_conditions(
           transverse_field_ising(Float64, Z2Irrep; J = 1.0, g = 1.0), L)
phys = Z2Space(0 => 1, 1 => 1)

# ─────────────────────────────────────────────
# Fit function
# ─────────────────────────────────────────────

"""
    estimate_c(ψ; L, fitcuts)

Fit the entanglement entropy profile of `ψ` to the PBC Cardy–Calabrese formula
and return `(c, slope, intercept, x, S)`.
"""
function estimate_c(ψ; L::Int, fitcuts)
    ℓs = 1:(L-1)
    S  = [entropy(ψ, ℓ) for ℓ in ℓs]
    x  = [log((L / π) * sin(π * ℓ / L)) for ℓ in ℓs]

    A     = hcat(ones(length(fitcuts)), x[fitcuts])
    coeff = A \ S[fitcuts]
    intercept, slope = coeff[1], coeff[2]

    return 3 * slope, slope, intercept, x, S
end

# ─────────────────────────────────────────────
# Run over bond dimensions
# ─────────────────────────────────────────────

fits = NamedTuple[]

for D in Ds
    bond    = Z2Space(0 => D ÷ 2, 1 => D ÷ 2)
    ψ0      = FiniteMPS(L, phys, bond)
    ψ, _, _ = find_groundstate(ψ0, H, DMRG())

    c, slope, intercept, x, S = estimate_c(ψ; L = L, fitcuts = fitcuts)
    push!(fits, (; D, c, slope, intercept, x, S))
    println("D = $D  →  c ≈ $(round(c; digits=6))")
end

# ─────────────────────────────────────────────
# Plot 1: entanglement entropy fit (largest D)
# ─────────────────────────────────────────────

best  = fits[end]
xfit  = range(minimum(best.x), maximum(best.x); length = 200)

p1 = scatter(best.x, best.S;
    label   = "DMRG (D=$(best.D))",
    xlabel  = "ln((L/π) sin(πℓ/L))",
    ylabel  = "S(ℓ)",
    color   = :maroon,
    legend  = :topleft,
    dpi     = 300,
)
plot!(p1, xfit, best.intercept .+ best.slope .* xfit;
    label     = "fit: c = $(round(best.c; digits=4)),  D=$(best.D),  L=$L",
    linewidth = 1,
    linestyle = :dash,
    color     = :black,
)
title!(p1, "Entanglement Entropy Scaling – PBC")

# ─────────────────────────────────────────────
# Plot 2: c vs bond dimension
# ─────────────────────────────────────────────

p2 = scatter([f.D for f in fits], [f.c for f in fits];
    label   = "c(D)",
    xlabel  = "Bond dimension D",
    ylabel  = "c",
    color   = :maroon,
    legend  = :bottomright,
    dpi     = 300,
)
hline!(p2, [0.5];
    label     = "Exact c = 1/2",
    linestyle = :dash,
    color     = :black,
)
title!(p2, "Central Charge vs Bond Dimension")

display(p1)
display(p2)
savefig(p1, "ising_c_fit.png")
savefig(p2, "ising_c_vs_D.png")
