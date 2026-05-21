using MPSKit, MPSKitModels, TensorKit, Plots
using Logging
global_logger(NullLogger())

# Choose system size and bond dimension
L = 12

# Build critical Ising Hamiltonian
model = transverse_field_ising(ComplexF64, Trivial; J=1.0, g=1.0)
H = periodic_boundary_conditions(model, L)

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