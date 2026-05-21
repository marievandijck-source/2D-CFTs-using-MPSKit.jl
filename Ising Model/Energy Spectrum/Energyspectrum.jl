using MPSKit, MPSKitModels, TensorKit, Plots, KrylovKit
using LinearAlgebra: eigvals, diagm, Hermitian
using Logging
global_logger(NullLogger())

using MPSKit, MPSKitModels, TensorKit

# Choose system size and bond dimension
L = 12

phys = Z2Space(0 => 1, 1 => 1)
bond = Z2Space(0 => D÷2, 1 => D÷2)

# Build critical Ising Hamiltonian
model = transverse_field_ising(Float64, Z2Irrep; J=1.0, g=1.0)
H = periodic_boundary_conditions(model, L)

# Initial MPS
ψ0 = FiniteMPS(L, phys, bond)

# Find 18 lowes energy values and states 
energies, states = exact_diagonalization(H; num = 18, alg = Lanczos(; krylovdim = 200));

plot(
    real.(energies);
    color=:maroon,
    seriestype = :scatter, legend = false, ylabel = "Energy", xlabel = "Number of the state",
    title="Energy spectrum",
    dpi=300
)

#savefig("Ising_energy.png")