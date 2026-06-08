#=
using MPSKit, MPSKitModels, TensorKit, Plots, KrylovKit
using LinearAlgebra: eigvals, diagm, Hermitian
using Logging
global_logger(NullLogger())


# Choose system size and bond dimension
L = 40
D = 50

phys = Z2Space(0 => 1, 1 => 1)
bond = Z2Space(0 => D÷2, 1 => D÷2)

# Build critical Ising Hamiltonian
model = transverse_field_ising(ComplexF64, Z2Irrep; J=1.0, g=1.0)
H = periodic_boundary_conditions(model, L)

# Initial MPS
ψ0 = FiniteMPS(L, phys, bond)

# Find ground state
ψ, envs, δ = find_groundstate(ψ0, H, DMRG())

# Rough thermodynamic energy density estimate
E0 = real(expectation_value(ψ, H))
e_inf_rough = E0 / L + π/(6 * L^2)

println("Rough estimate e_inf ≈ $e_inf_rough")



function energy_density_ising(L, D)
    phys = Z2Space(0 => 1, 1 => 1)
    bond = Z2Space(0 => D÷2, 1 => D÷2)

    model = transverse_field_ising(ComplexF64, Z2Irrep; J=1.0, g=1.0)
    H = periodic_boundary_conditions(model, L)

    ψ0 = FiniteMPS(L, phys, bond)
    ψ, _, _ = find_groundstate(ψ0, H, DMRG())

    E0 = real(expectation_value(ψ, H))
    println(E0 / L + π/(6 * L^2))
    return E0 / L + π/(6 * L^2)
end


Ls = 5:2:31
e_vals = [energy_density_ising(L, D) for L in Ls]

println(e_vals)

scatter(
    Ls,
    e_vals,
    xlabel="L",
    ylabel="e∞(L)",
    label="DMRG",
    color=:maroon,
    title="Bulk free energy e∞(L)",
    dpi=300
)

hline!([-4/π], linestyle=:dash, label="Exact e∞", color=:black)

#savefig("Ising_bfe.png")
=#

using MPSKit, MPSKitModels, TensorKit, Plots, KrylovKit
using LinearAlgebra: eigvals, diagm, Hermitian
using Logging
global_logger(NullLogger())


function gs(L, D)
    phys = Z2Space(0 => 1, 1 => 1)
    bond = Z2Space(0 => D÷2, 1 => D÷2)

    model = transverse_field_ising(ComplexF64, Z2Irrep; J=1.0, g=1.0)
    H = periodic_boundary_conditions(model, L)

    ψ0 = FiniteMPS(L, phys, bond)
    ψ, _, _ = find_groundstate(ψ0, H, DMRG())

    E0 = real(expectation_value(ψ, H))

    return E0
end

D = 40
Ls = 20:2:30
e_0 = [gs(L, D) for L in Ls]

x = 1 ./ Ls.^2
A = hcat(ones(length(Ls)), x)
coeff = A \ (e_0 ./ Ls)
a, b = coeff[1], -coeff[2]

println("e_inf = $(round(a, digits=6))")
println("π v / 6 = $(round(b, digits=6))")
println("v = $(round(12b/π, digits=6))")