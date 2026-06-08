using MPSKit, MPSKitModels, TensorKit, Plots, KrylovKit
using LinearAlgebra: eigvals, diagm, Hermitian
using Pkg
using Logging
global_logger(NullLogger())

include("Functions.jl")

L = 4
D = 10
n = 1

#critical Ising Hamiltonian
H = periodic_boundary_conditions(heisenberg_hamiltonian(), L)
energies, ψ, sectors = DMRG_H(H, L, D, n)
#Rough thermodynamic energy density estimate
E0 = real(expectation_value(ψ[1], H))
energies, states = exact_diagonalization(H; num = 18, alg = Lanczos(; krylovdim = 200));
println("okayyy")
function energy_density_heisenberg(L, D)

    H = periodic_boundary_conditions(heisenberg_hamiltonian(), L)

    n = 1
    energies, ψ, sectors = DMRG_H(H, L, D, n)

    E0 = real(expectation_value(ψ[1], H))
    println(L)
    return E0
end

Ls = 12:4:44
D = 70
#p = 12, Ls = 40, D = 50 → -0.443145188052332 and 1.5816844813692137
#p = 15, Ls = 44, D = 60 → -0.44314595009041485 and 1.5800165339915113
e_vals = [energy_density_heisenberg(L, D) for L in Ls]

Lvals = Float64.(Ls)

X = hcat(Lvals, 1.0 ./ Lvals)

p = 15
w = Lvals .^ p

Xw = X .* sqrt.(w)
yw = e_vals .* sqrt.(w)

β = Xw \ yw
a = β[1]
b = β[2]

println(a)
println(-b*6/π)