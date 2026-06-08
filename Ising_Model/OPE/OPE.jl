using MPSKit, MPSKitModels, TensorKit, Plots
using LinearAlgebra, KrylovKit
using Logging
global_logger(NullLogger())


# Check eerst wat de structuur is van Z
# Maak expliciet een twee-site operator

L = 7
D = 20

println("D=$D")
println("L=$L")
phys = Z2Space(0 => 1, 1 => 1)
bond = Z2Space(0 => D÷2, 1 => D÷2)

# Critical Ising Hamiltonian
model = transverse_field_ising(Float64, Trivial; J=1.0, g=1.0)
H = periodic_boundary_conditions(model, L)

ψ0 = FiniteMPS(L, ℂ^2, ℂ^D)
ψ, _, _ = find_groundstate(ψ0, H, DMRG())

#energies, ψ = exact_diagonalization(H; num = 1, alg = Lanczos(; krylovdim = 200));

# Gewone Pauli Z matrix
Z = TensorMap(Float64[1 0; 0 -1], ℂ^2, ℂ^2)

# Test single site
@show expectation_value(ψ, 1 => Z)

# Test two-site
@show expectation_value(ψ, (1, 2) => Z ⊗ Z)


function extract_OPE(ψ, L)
    mid = L ÷ 2
    r   = L ÷ 4
    i, j, k = mid - r, mid + r, mid

    # True three-point: <Z_i Z_j Z_k Z_{k+1}>
    # Use two separate two-site operators at different separations
    G3 = expectation_value(ψ, (i, k, k+1, j) => Z ⊗ Z ⊗ Z ⊗ Z)

    # Two-point σσ: <Z_i Z_j>
    G2_sigma = expectation_value(ψ, (i, j) => Z ⊗ Z)

    # Two-point εε: <Z_k Z_{k+1}>  (energy density at one point)
    G2_eps = expectation_value(ψ, (k, k+1) => Z ⊗ Z)

    C = real(G3) / sqrt(real(G2_sigma)^2 * real(G2_eps))

    return C
end

C = extract_OPE(ψ, L)
println("C_σσε ≈ $(round(C, digits=4))  (exact: 0.5)")
