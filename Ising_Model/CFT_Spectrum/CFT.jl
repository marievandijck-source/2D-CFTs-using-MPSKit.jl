using MPSKit, MPSKitModels, TensorKit, Plots
using LinearAlgebra: eigvals, diagm, Hermitian

# Shift operator for momentum measurement
function translation_operator(L)
    I = id(ComplexF64, ℂ^2)
    @tensor O[W S; N E] := I[W; N] * I[S; E]
    return periodic_boundary_conditions(InfiniteMPO([O]), L)
end

# Fix degeneracies for a group of states
function momenta_from_states(states; tol=1e-10)
    L = length(states[1])
    T = translation_operator(L)
    n = length(states)
    
    # Build overlap matrix
    M = [dot(states[i], T, states[j]) for i in 1:n, j in 1:n]
    
    return angle.(eigvals(M))
end

# Detect degenerate groups automatically
function group_degenerate_states(energies; tol=1e-6)
    groups = Vector{UnitRange{Int}}()
    i = 1
    while i <= length(energies)
        j = i
        while j < length(energies) && abs(energies[j+1] - energies[i]) < tol
            j += 1
        end
        push!(groups, i:j)
        i = j + 1
    end
    return groups
end

# Full pipeline
function compute_spectrum(L, D; num=23)
    H = periodic_boundary_conditions(transverse_field_ising(), L)

    ψ, envs, _ = find_groundstate(FiniteMPS(L, ℂ^2, ℂ^D), H, DMRG())
    E_ex, qps = excitations(H, QuasiparticleAnsatz(), ψ, envs; num=num-1)

    states = vcat(ψ, map(qp -> convert(FiniteMPS, qp), qps))
    energies = real.(map(x -> expectation_value(x, H), states))

    # Detect and fix degenerate groups
    groups = group_degenerate_states(energies)
    momenta = Float64[]
    for g in groups
        append!(momenta, momenta_from_states(states[g]))
    end

    v = 2.0
    Δ = (energies .- energies[1]) ./ (2π * v / L)
    S = momenta ./ (2π / L)

    return S, Δ
end

# Run and plot
L, D = 24, 70
S, Δ = compute_spectrum(L, D)

p = scatter(S, Δ;
    xlabel="Conformal spin (S)",
    ylabel="Scaling dimension (Δ)",
    legend=true,
    label="",
    color=:maroon,
    dpi=300)

scatter!(p, S[1:3], Δ[1:3];
    label = "Primaries",
    color = :firebrick1,
    markersize = 6)

tol = 0.1

mask_T = (abs.(Δ .- 2) .< tol) .&
         (abs.(abs.(S) .- 2) .< tol)

scatter!(p, S[mask_T], Δ[mask_T] , label = "Stress–energy tensor",  color = :maroon, markershape = :diamond, markersize = 7)





vline!(p, -3:3; color=:gray, linestyle=:dash, label="")
hline!(p, [0, 1/8, 1, 9/8, 2, 17/8, 3, 25/8]; color=:gray, linestyle=:dash, label="")
title!(p, "CFT Spectrum — Ising, L=$L")
display(p)