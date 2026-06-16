using MPSKit, MPSKitModels, TensorKit, Plots
using Logging
global_logger(NullLogger())

# ─────────────────────────────────────────────
# Raw energy spectrum — critical Ising chain (ED)
# ─────────────────────────────────────────────

L = 12

H = periodic_boundary_conditions(
        transverse_field_ising(ComplexF64, Trivial; J = 1.0, g = 1.0), L)

energies, _ = exact_diagonalization(H; num = 18, alg = Lanczos(; krylovdim = 200))

scatter(real.(energies);
    ylabel    = "Energy",
    xlabel    = "State index",
    color     = :maroon,
    legend    = false,
    title     = "Energy spectrum – Ising, L=$L",
    dpi       = 300,
)
