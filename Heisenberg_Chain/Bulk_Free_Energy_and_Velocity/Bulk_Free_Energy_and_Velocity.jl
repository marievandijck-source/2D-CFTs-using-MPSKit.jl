using MPSKit, MPSKitModels, TensorKit, Plots, KrylovKit
using Logging
global_logger(NullLogger())

include("Functions.jl")

# ─────────────────────────────────────────────
# Bulk free energy density and velocity
# Heisenberg spin-1/2 chain
# ─────────────────────────────────────────────

"""
    energy_density_heisenberg(L, D)

Compute the ground-state energy density E₀/L for a Heisenberg chain
of length `L` with DMRG bond dimension `D`.
"""
function energy_density_heisenberg(L, D)
    H = periodic_boundary_conditions(heisenberg_hamiltonian(), L)
    energies, ψ, _ = DMRG_H(H, L, D, [1])
    E0 = real(sum(expectation_value(ψ[1], H)))
    println("L = $L  →  e₀ = $(E0 / L)")
    return E0
end

# ─────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────

Ls = 12:4:44   # system sizes
D  = 70        # bond dimension

# ─────────────────────────────────────────────
# Collect ground-state energies
# ─────────────────────────────────────────────

e_vals = [energy_density_heisenberg(L, D) for L in Ls]

# ─────────────────────────────────────────────
# Finite-size extrapolation
#
# Ansatz:  E₀(L) = a·L + b/L
#   → e∞ = a  (bulk energy density)
#   → c  = -b·6/π  (central charge, via CFT finite-size formula)
# ─────────────────────────────────────────────

p     = 15                          # power-law weight exponent
Lvals = Float64.(collect(Ls))
X     = hcat(Lvals, 1.0 ./ Lvals)  # design matrix [L, 1/L]
w     = Lvals .^ p                  # weights favour larger L

β = (X .* sqrt.(w)) \ (e_vals .* sqrt.(w))   # weighted least squares
a, b = β[1], β[2]

e_bulk   = a
c_from_b = -b * 6 / π

println("\n── Extrapolation results ──")
println("Bulk energy density e∞ = $e_bulk")
println("Central charge c       = $c_from_b")
