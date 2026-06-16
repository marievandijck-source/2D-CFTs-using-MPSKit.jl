using MPSKit, MPSKitModels, TensorKit, Plots
using LinearAlgebra: eigvals
using Logging
global_logger(NullLogger())

# ─────────────────────────────────────────────
# CFT spectrum — critical Ising chain
#
# Scaling dimensions:  Δ = (E - E₀) / (2πv/L),  v = 2
# Expected primaries:  𝟙 (Δ=0), σ (Δ=1/8), ε (Δ=1)
# ─────────────────────────────────────────────

"""
    translation_operator(L)

Return the one-site translation MPO on a ring of length `L` (spinless / ℂ² physical space).
"""
function translation_operator(L)
    I_op = id(ComplexF64, ℂ^2)
    @tensor O[W S; N E] := I_op[W; N] * I_op[S; E]
    return periodic_boundary_conditions(InfiniteMPO([O]), L)
end

"""
    momenta_from_states(states)

Diagonalize the translation operator within a degenerate subspace and return momenta.
"""
function momenta_from_states(states)
    T = translation_operator(length(states[1]))
    M = [dot(states[i], T, states[j]) for i in eachindex(states), j in eachindex(states)]
    return angle.(eigvals(M))
end

"""
    group_degenerate(energies; atol = 1e-6)

Return index ranges of degenerate energy groups.
"""
function group_degenerate(energies; atol = 1e-6)
    groups = UnitRange{Int}[]
    i = 1
    while i <= length(energies)
        j = i
        while j < length(energies) && abs(energies[j+1] - energies[i]) < atol
            j += 1
        end
        push!(groups, i:j)
        i = j + 1
    end
    return groups
end

"""
    compute_spectrum(L, D; num = 23)

Run DMRG + QuasiparticleAnsatz for the critical Ising chain and return
conformal spins `S` and scaling dimensions `Δ`.
"""
function compute_spectrum(L, D; num = 23)
    H       = periodic_boundary_conditions(transverse_field_ising(), L)
    ψ, envs, _ = find_groundstate(FiniteMPS(L, ℂ^2, ℂ^D), H, DMRG())
    _, qps  = excitations(H, QuasiparticleAnsatz(), ψ, envs; num = num - 1)

    states    = vcat([ψ], map(qp -> convert(FiniteMPS, qp), qps))
    energies  = real.(map(x -> expectation_value(x, H), states))

    momenta = Float64[]
    for g in group_degenerate(energies)
        append!(momenta, momenta_from_states(states[g]))
    end

    v = 2.0
    Δ = (energies .- energies[1]) ./ (2π * v / L)
    S = momenta ./ (2π / L)
    return S, Δ
end

# ─────────────────────────────────────────────
# Parameters and run
# ─────────────────────────────────────────────

L, D = 24, 70
S, Δ  = compute_spectrum(L, D)

# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────

p = scatter(S, Δ;
    xlabel        = "Conformal spin S",
    ylabel        = "Scaling dimension Δ",
    label         = "",
    color         = :maroon,
    markerstrokewidth = 0.5,
    framestyle    = :box,
    dpi           = 300,
)

# Highlight primary operators (three lowest states)
scatter!(p, S[1:3], Δ[1:3];
    label      = "Primaries",
    color      = :firebrick1,
    markersize = 6,
)

# Highlight stress–energy tensor (Δ ≈ 2, |S| ≈ 2)
mask_T = (abs.(Δ .- 2) .< 0.1) .& (abs.(abs.(S) .- 2) .< 0.1)
scatter!(p, S[mask_T], Δ[mask_T];
    label         = "Stress–energy tensor",
    color         = :maroon,
    markershape   = :diamond,
    markersize    = 7,
)

# Reference lines at expected scaling dimensions
vline!(p, -3:3;           color = :gray, linestyle = :dash, label = "")
hline!(p, [0, 1/8, 1, 9/8, 2, 17/8, 3, 25/8];
           color = :gray, linestyle = :dash, label = "")

title!(p, "CFT Spectrum – Ising, L=$L")
display(p)
