using MPSKit, MPSKitModels, TensorKit, KrylovKit
using LinearAlgebra: eigvals

# ─────────────────────────────────────────────
# Hamiltonian
# ─────────────────────────────────────────────

casimir(s::SU2Irrep) = s.j * (s.j + 1)

"""
    heisenberg_hamiltonian(; J = -1.0)

Return the SU(2)-symmetric Heisenberg Hamiltonian as an `InfiniteMPOHamiltonian`.
`J < 0` is antiferromagnetic.
"""
function heisenberg_hamiltonian(; J = -1.0)
    s  = SU2Irrep(1//2)
    ℋ  = SU2Space(s => 1)
    SS = zeros(ComplexF64, ℋ ⊗ ℋ ← ℋ ⊗ ℋ)
    for (S, data) in blocks(SS)
        data .= -0.5J * (casimir(S) - casimir(s) - casimir(s))
    end
    return InfiniteMPOHamiltonian(SS)
end

# ─────────────────────────────────────────────
# Bond space
# ─────────────────────────────────────────────

"""
    bondspace_heis(D; jmax = 6)

Construct an SU(2)-symmetric bond space with approximate total dimension `D`,
distributing multiplicities across spin-j representations with a Gaussian envelope.
"""
function bondspace_heis(D; jmax = 6)
    reps = Pair{Rational{Int}, Int}[]
    a = 0.4
    for twoj in 0:(2 * jmax)
        j    = twoj // 2
        mult = max(1, round(Int, D * exp(-a * j^2)))
        push!(reps, j => mult)
    end
    return SU2Space(reps...)
end

# ─────────────────────────────────────────────
# Exact diagonalization
# ─────────────────────────────────────────────

"""
    ED_H(H, n)

Run exact diagonalization of `H` across SU(2) sectors S = 0, 1/2, 1, …
`n[rep]` states are requested per sector. Returns energies, states, and sector labels
sorted by energy.
"""
function ED_H(H, n)
    all_energies = []
    all_states   = []
    all_sectors  = []

    for rep in 1:length(n)
        S_val = rep - 1
        ener, st = exact_diagonalization(
            H; num = n[rep], sector = SU2Irrep(rep - 0.5),
            alg = Lanczos(; krylovdim = 300)
        )
        append!(all_energies, ener[1:n[rep]])
        append!(all_states,   st[1:n[rep]])
        append!(all_sectors,  fill(S_val, n[rep]))
        println("Sector S = $S_val done")
    end

    p_idx = sortperm(real.(all_energies))
    return all_energies[p_idx], all_states[p_idx], all_sectors[p_idx]
end

# ─────────────────────────────────────────────
# DMRG
# ─────────────────────────────────────────────

"""
    DMRG_H(H, L, D, n)

Find the ground state of `H` on a chain of length `L` with bond dimension `D` via DMRG,
then compute low-lying excitations across SU(2) sectors. Returns energies, states, and
sector labels sorted by energy.
"""
function DMRG_H(H, L, D, n)
    d      = SU2Space(1//2 => 1)
    V_edge = SU2Space(0 => 1)
    V      = bondspace_heis(D)

    ψ0         = FiniteMPS(L, d, V; left = V_edge, right = V_edge)
    ψ_gs, _, _ = find_groundstate(ψ0, H, DMRG(; maxiter = 10, tol = 1e-6))

    all_energies = []
    all_states   = []
    all_sectors  = []

    for rep in 1:length(n)
        S_val = rep - 1
        _, st = excitations(
            H, QuasiparticleAnsatz(; krylovdim = 50), ψ_gs;
            sector = SU2Irrep(S_val), num = n[rep]
        )
        mps_states = map(qp -> convert(FiniteMPS, qp), st)

        # Include the ground state in the S = 0 sector
        current_states   = (rep == 1) ? vcat([ψ_gs], mps_states) : mps_states
        current_energies = map(x -> sum(expectation_value(x, H)), current_states)

        append!(all_energies, current_energies)
        append!(all_states,   current_states)
        append!(all_sectors,  fill(S_val, length(current_energies)))
        println("Sector S = $S_val done")
    end

    p_idx = sortperm(real.(all_energies))
    return all_energies[p_idx], all_states[p_idx], all_sectors[p_idx]
end

# ─────────────────────────────────────────────
# Momentum / translation operator
# ─────────────────────────────────────────────

"""
    O_shift(L)

Return the one-site translation MPO on a ring of length `L`.
"""
function O_shift(L)
    phys = SU2Space(1//2 => 1)
    I_op = id(ComplexF64, phys)
    @tensor O[W S; N E] := I_op[W; N] * I_op[S; E]
    return periodic_boundary_conditions(InfiniteMPO([O]), L)
end

"""
    fix_degeneracies(basis)

Diagonalize the translation operator within a degenerate subspace spanned by `basis`
and return the resulting momenta (angles of eigenvalues).
"""
function fix_degeneracies(basis)
    L = length(basis[1])
    T = O_shift(L)
    M = zeros(ComplexF64, length(basis), length(basis))
    for j in eachindex(basis), i in eachindex(basis)
        M[i, j] = dot(basis[i], T, basis[j])
    end
    vals = eigvals(M)
    # Normalize ±π to +π to avoid floating-point sign ambiguity
    return map(vals) do λ
        θ = angle(λ)
        isapprox(abs(θ), π; atol = 1e-5) ? π : θ
    end
end

"""
    compute_momenta(energies, states)

Group degenerate states (within `atol = 1e-4`) and diagonalize the translation operator
within each group to assign momenta.

`compute_momenta_mps` is an alias for use with FiniteMPS states.
"""
function compute_momenta(energies, states)
    momenta = Float64[]
    i = 1
    while i <= length(energies)
        j = i
        while j <= length(energies) && isapprox(energies[j], energies[i]; atol = 1e-4)
            j += 1
        end
        append!(momenta, fix_degeneracies(states[i:j-1]))
        i = j
    end
    return momenta
end

const compute_momenta_mps = compute_momenta

# ─────────────────────────────────────────────
# Velocity estimator
# ─────────────────────────────────────────────

"""
    estimate_v(S, sectors, energies, L)

Estimate the spinon velocity from the spectrum using the lowest S = 0 states near
momentum 0 (ground state) and near momentum ±2π (first excited tower).
"""
function estimate_v(S, sectors, energies, L)
    E_zero  = Float64[]
    E_tower = Float64[]

    for i in eachindex(S)
        if sectors[i] == 0
            if -0.1 <= S[i] <= 0.1
                push!(E_zero,  real(energies[i]))
            elseif 1.8 <= abs(S[i]) <= 2.2
                push!(E_tower, real(energies[i]))
            end
        end
    end

    return (minimum(E_tower) - minimum(E_zero)) * L / (4π)
end
