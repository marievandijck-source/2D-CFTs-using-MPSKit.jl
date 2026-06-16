using MPSKit, MPSKitModels, TensorKit, Plots, KrylovKit
using LinearAlgebra
using Logging
global_logger(NullLogger())

# ─────────────────────────────────────────────
# Bulk free energy density — critical Ising chain
#
# Finite-size ansatz:  E₀(L)/L = e∞ + (πv/6) · (1/L²)
# Exact value:  e∞ = −4/π ≈ −1.2732
# ─────────────────────────────────────────────

"""
    gs_energy(L, D)

Return the DMRG ground-state energy for a Z₂-symmetric Ising chain
of length `L` with bond dimension `D` (PBC).
"""
function gs_energy(L, D)
    phys  = Z2Space(0 => 1, 1 => 1)
    bond  = Z2Space(0 => D ÷ 2, 1 => D ÷ 2)
    H     = periodic_boundary_conditions(
                transverse_field_ising(ComplexF64, Z2Irrep; J = 1.0, g = 1.0), L)
    ψ0    = FiniteMPS(L, phys, bond)
    ψ, _, _ = find_groundstate(ψ0, H, DMRG())
    return real(expectation_value(ψ, H))
end

# ─────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────

D  = 40
Ls = 20:2:30

# ─────────────────────────────────────────────
# Collect energies and extrapolate
# ─────────────────────────────────────────────

E0 = [gs_energy(L, D) for L in Ls]

# Weighted least squares:  E₀/L = e∞ + b/L²
A     = hcat(ones(length(Ls)), 1.0 ./ Ls .^ 2)
coeff = A \ (E0 ./ Ls)
e_inf, b = coeff[1], coeff[2]

println("e∞       = $(round(e_inf;    digits=6))  (exact: $(round(-4/π; digits=6)))")
println("πv/6     = $(round(b;        digits=6))")
println("v        = $(round(6b / π;   digits=6))  (exact: 2.0)")
