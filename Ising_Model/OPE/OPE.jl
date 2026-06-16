using MPSKit, MPSKitModels, TensorKit, Plots
using LinearAlgebra, KrylovKit
using Logging
global_logger(NullLogger())

# ─────────────────────────────────────────────
# OPE coefficient C_σσε — critical Ising chain
#
# Estimated via the three-point function of the order parameter σ ~ Z,
# using the ground state from DMRG.
# Exact value:  C_σσε = 1/2
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────

L = 7
D = 20

# ─────────────────────────────────────────────
# Ground state
# ─────────────────────────────────────────────

H       = periodic_boundary_conditions(
              transverse_field_ising(Float64, Trivial; J = 1.0, g = 1.0), L)
ψ0      = FiniteMPS(L, ℂ^2, ℂ^D)
ψ, _, _ = find_groundstate(ψ0, H, DMRG())

# Pauli-Z operator (order parameter σ)
Z = TensorMap(Float64[1 0; 0 -1], ℂ^2, ℂ^2)

# ─────────────────────────────────────────────
# OPE coefficient extraction
#
# C_σσε ≈ ⟨Z_i Z_j Z_k Z_{k+1}⟩ / sqrt(⟨Z_i Z_j⟩² · ⟨Z_k Z_{k+1}⟩)
# with i, j separated by L/2 and k at the midpoint.
# ─────────────────────────────────────────────

"""
    extract_OPE(ψ, L)

Estimate the OPE coefficient C_σσε from the ground state `ψ` of a chain of length `L`.
"""
function extract_OPE(ψ, L)
    mid = L ÷ 2
    r   = L ÷ 4
    i, j, k = mid - r, mid + r, mid

    G4      = expectation_value(ψ, (i, k, k+1, j) => Z ⊗ Z ⊗ Z ⊗ Z)
    G2_σ    = expectation_value(ψ, (i, j)         => Z ⊗ Z)
    G2_ε    = expectation_value(ψ, (k, k+1)       => Z ⊗ Z)

    return real(G4) / sqrt(real(G2_σ)^2 * real(G2_ε))
end

C = extract_OPE(ψ, L)
println("C_σσε ≈ $(round(C; digits=4))  (exact: 0.5)")
