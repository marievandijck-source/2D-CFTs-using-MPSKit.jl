casimir(s::SU2Irrep) = s.j * (s.j + 1)

function heisenberg_hamiltonian(; J = -1.0)
    s = SU2Irrep(1//2)
    ℋ = SU2Space(s => 1)
    SS = zeros(ComplexF64, ℋ ⊗ ℋ ← ℋ ⊗ ℋ)
    for (S, data) in blocks(SS)
        data .= -0.5J * (casimir(S) - casimir(s) - casimir(s))
    end
    return InfiniteMPOHamiltonian(SS)
end

function estimate_v(S)
    E = []
    T = []

    for i in 1:length(S)
        if -0.1 <= S[i] && S[i] <= 0.1
            if sectors[i] == 0
                push!(E,real.(energies[i]))
            end
        elseif 1.8 <= abs(S[i]) && abs(S[i]) <= 2.2
            if sectors[i] == 0
                push!(T, real.(energies[i]))
            end
        end 
    end

    ET = minimum(T)
    E0 = minimum(E)


    v = (ET - E0) * L / (4π)
    return v
end

function sort(energies_0, energies_1, all_energies, all_states)
    all_sectors = vcat(fill(0, length(energies_0)), fill(1, length(energies_1)))
    p_idx = sortperm(real.(all_energies))
    energies = all_energies[p_idx]
    states = all_states[p_idx]
    sectors = all_sectors[p_idx]
return energies, states, sectors
end

function ED_H(H, n)

    all_energies = []
    all_states = []
    all_sectors = []

    for rep in 1:length(n)
        ener, st = exact_diagonalization(H; num=n[rep], sector=SU2Irrep(rep-0.5), alg=Lanczos(; krylovdim=300))
        ener = ener[1:n[rep]]
        st = st[1:n[rep]]
        append!(all_energies, ener)
        append!(all_states, st)
        append!(all_sectors, fill(rep - 1, length(ener))) # S=0, S=1, etc.
        println("Sector S = $(rep-1) voltooid")
    end
    p_idx = sortperm(real.(all_energies))
    
    return all_energies[p_idx], all_states[p_idx], all_sectors[p_idx]
end

function compute_momenta(energies, states)
    number = length(energies)
    momenta = Float64[]
    i = 1
    while i <= number
        j = i
        while j <= number && j <= length(energies) && isapprox(energies[j], energies[i]; atol=1e-4)
            j += 1
        end
        group_momenta = fix_degeneracies(states[i:j-1])
        append!(momenta, group_momenta)
        i = j
    end
    return momenta
end

function O_shift(L)
    phys = SU2Space(1//2 => 1)
    I = id(ComplexF64, phys)
    #MPO for identityoperator 
    @tensor O[W S; N E] := I[W; N] * I[S; E]
    #connect input from stite i to site i+1
    return periodic_boundary_conditions(InfiniteMPO([O]), L)
end

function fix_degeneracies(basis)
    L = length(basis[1])
    M = zeros(ComplexF64, length(basis), length(basis))
    T = O_shift(L)
    for j in eachindex(basis), i in eachindex(basis)
        M[i, j] = dot(basis[i], T, basis[j])
    end

    vals = eigvals(M)
    angles = angle.(vals)
    
    # Normalize ±π to +π to avoid sign ambiguity from floating point noise
    angles = map(angles) do θ
        isapprox(abs(θ), π; atol=1e-5) ? π : θ
    end
    return angle.(vals)
end

function DMRG_H(H, L, D, n) 
    
    d = SU2Space(1//2 => 1)
    V_edgel = SU2Space(0 => 1) 
    V_edger = SU2Space(0 => 1)
    V = bondspace_heis(D)
    ψ0 = FiniteMPS(L, d, V; left = V_edgel, right=V_edger)
    
    alg = DMRG(; maxiter=10, tol=1e-6)
    ψ_gs, envs, _ = find_groundstate(ψ0, H, alg)
    
    all_energies = []
    all_states = []
    all_sectors = []
    

    for rep in 1:length(n)
        # 1. Haal de excitaties op voor de specifieke SU2 sector
        # We gebruiken tijdelijke variabelen 'ener' en 'st'
        
        ener, st = excitations(H, QuasiparticleAnsatz(; krylovdim = 50), ψ_gs; 
                               sector = SU2Irrep(rep - 1), num = n[rep])
        # 2. Converteer Quasiparticles naar FiniteMPS
        mps_states = map(qp -> convert(FiniteMPS, qp), st)
        # 3. Voeg de Grondtoestand toe als we in de S=0 sector zijn (rep=1)
        if rep == 1
            current_states = vcat([ψ_gs], mps_states)
        else
            current_states = mps_states
        end
        
        # 4. Bereken de verwachtingswaarden van de energie
        current_energies = map(x -> sum(expectation_value(x, H)), current_states)
        
        # 5. Voeg alles toe aan de verzamellijsten
        append!(all_energies, current_energies)
        append!(all_states, current_states)
        append!(all_sectors, fill(rep - 1, length(current_energies))) # S=0, S=1, etc.
        
        println("Sector S = $(rep-1) voltooid")
    end

    # 6. Sorteren op basis van energie
    p_idx = sortperm(real.(all_energies))
    return all_energies[p_idx], all_states[p_idx], all_sectors[p_idx]
end

function compute_momenta_mps(energies, states)
    number = length(energies)
    momenta = Float64[]
    i = 1
    while i <= number
        j = i
        while j <= number && j <= length(energies) && isapprox(energies[j], energies[i]; atol=1e-4)
            j += 1
        end
        group_momenta = fix_degeneracies_mps(states[i:j-1])
        append!(momenta, group_momenta)
        i = j
    end
    return momenta
end

function O_shift_mps(L)
    phys = SU2Space(1//2 => 1)
    I = id(ComplexF64, phys)
    #MPO for identityoperator 
    @tensor O[W S; N E] := I[W; N] * I[S; E]
    #connect input from stite i to site i+1
    return periodic_boundary_conditions(InfiniteMPO([O]), L)
end

function fix_degeneracies_mps(basis)
    L = length(basis[1])
    M = zeros(ComplexF64, length(basis), length(basis))
    T = O_shift(L)
    for j in eachindex(basis), i in eachindex(basis)
        M[i, j] = dot(basis[i], T, basis[j])
    end

    vals = eigvals(M)
    angles = angle.(vals)
    
    # Normalize ±π to +π to avoid sign ambiguity from floating point noise
    angles = map(angles) do θ
        isapprox(abs(θ), π; atol=1e-5) ? π : θ
    end
    return angle.(vals)
end

function bondspace_heis(D; jmax=6)
    reps = Pair{Rational{Int},Int}[]
    for twoj in 0:(2*jmax)
        j = twoj//2
        a = 0.4
        mult = max(1, round(Int, D * exp(-a * j^2)))
        push!(reps, j => mult)
    end
    return SU2Space(reps...)
end
