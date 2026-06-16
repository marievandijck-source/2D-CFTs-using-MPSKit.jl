using MPSKit, MPSKitModels, TensorKit, KrylovKit
using LinearAlgebra, Plots
using Logging
global_logger(NullLogger())

include("Functions.jl")

# ─────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────

L   = 140
Ds  = [200]

# ─────────────────────────────────────────────
# Hamiltonian (open boundary conditions)
# ─────────────────────────────────────────────

H    = open_boundary_conditions(heisenberg_hamiltonian(), L)
phys = SU2Space(1//2 => 1)

# ─────────────────────────────────────────────
# Central charge from entanglement entropy (OBC)
#
# Cardy–Calabrese formula (OBC):
#   S(ℓ) = (c/6) log[(2L/π) sin(πℓ/L)] + const
# ─────────────────────────────────────────────

"""
    estimate_c_from_state_OBC(ψ; L, fitcuts)

Fit the entanglement entropy profile of `ψ` to the OBC Cardy–Calabrese formula
and return the estimated central charge `c` along with fit details.
"""
function estimate_c_from_state_OBC(ψ; L::Int, fitcuts = 8:(L-8))
    ℓs = collect(1:L-1)
    S  = [entropy(ψ, ℓ) for ℓ in ℓs]
    x  = log.((2L / π) .* sin.(π .* ℓs ./ L))

    xf, yf = x[fitcuts], S[fitcuts]
    coeff   = hcat(ones(length(xf)), xf) \ yf
    intercept, slope = coeff

    c_est = 6 * slope
    return c_est, slope, intercept, ℓs, S, x
end

# ─────────────────────────────────────────────
# Run over bond dimensions
# ─────────────────────────────────────────────

fits = []

for D in Ds
    ψ0      = FiniteMPS(L, phys, bondspace_heis(D))
    ψ, _, _ = find_groundstate(ψ0, H, DMRG())

    c_est, slope, intercept, ℓs, S, x = estimate_c_from_state_OBC(
        ψ; L = L, fitcuts = 4:2:(L-4)
    )

    push!(fits, (; c_est, slope, intercept, ℓs, S, x))
    println("D = $D  →  c ≈ $(round(c_est; digits=6))")
end

# ─────────────────────────────────────────────
# Plot best result
# ─────────────────────────────────────────────

bf     = fits[end]
yfit   = bf.intercept .+ bf.slope .* bf.x
even_i = findall(iseven, bf.ℓs)
odd_i  = findall(isodd,  bf.ℓs)

scatter(bf.x[even_i], bf.S[even_i];
    label   = "even ℓ",
    color   = :maroon,
    xlabel  = "ln((2L/π) sin(πℓ/L))",
    ylabel  = "S(ℓ)",
    dpi     = 300,
)
scatter!(bf.x[odd_i], bf.S[odd_i];
    label = "odd ℓ",
    color = :firebrick1,
)
plot!(bf.x, yfit;
    label     = "fit: c = $(round(bf.c_est; digits=4))",
    linewidth = 1.1,
    linestyle = :dash,
    color     = :black,
)
title!("Entanglement Entropy Scaling: OBC")
