using LinearAlgebra, Statistics
using KrylovKit
using LaTeXStrings
using MPSKit, MPSKitModels, TensorKit
using Plots
using Logging
global_logger(NullLogger())

include("C:/Users/marie/OneDrive/Bureaublad/UGent/thesis/Julia/Heisenberg_JOB/Functions.jl")

Ds = [80]
L = 48 #must be 4*n

model = heisenberg_hamiltonian()
H = periodic_boundary_conditions(model, L)   # PBC

phys = SU2Space(1//2 => 1)

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

function estimate_c_from_state_PBC(ψ; L::Int, fitcuts=4:(L-4))
    ℓs = collect(1:L-1)

    S = [entropy(ψ, ℓ) for ℓ in ℓs]

    x = log.((L / π) .* sin.(π .* ℓs ./ L))

    xf = x[fitcuts]
    yf = S[fitcuts]

    M = hcat(ones(length(xf)), xf)

    coeff = M \ yf
    intercept, slope = coeff

    c_est = 3 * slope   # PBC: S = c/3 log(...) + const

    return c_est, slope, intercept, ℓs, S, x
end

cs = Float64[]
states = Any[]
fits = Any[]

for D in Ds
    bond = bondspace_heis(D; jmax=6)

    ψ0 = FiniteMPS(L, phys, bond)

    ψ, env, δ = find_groundstate(ψ0, H, DMRG())

    c_est, slope, intercept, ℓs, S, x = estimate_c_from_state_PBC(
        ψ;
        L=L,
        fitcuts=4:(L-4)
    )

    push!(cs, c_est)
    push!(states, ψ)
    push!(fits, (
        c_est=c_est,
        slope=slope,
        intercept=intercept,
        ℓs=ℓs,
        S=S,
        x=x
    ))

    println("D = $D -> c ≈ $(round(c_est, digits=6))")
end

bestfit = fits[end]
bestD = Ds[end]

ℓs = bestfit.ℓs
S = bestfit.S
xvals = bestfit.x
slope = bestfit.slope
intercept = bestfit.intercept
c_best = bestfit.c_est

yfit = intercept .+ slope .* xvals

even_idx = findall(iseven, ℓs)
odd_idx  = findall(isodd, ℓs)

scatter(
    xvals[even_idx],
    S[even_idx],
    label="even ℓ",
    xlabel="ln((L/π) sin(πℓ/L))",
    ylabel="S(ℓ)",
    color = :maroon,
    dpi=300
)

scatter!(
    xvals[odd_idx],
    S[odd_idx],
    label="odd ℓ",
    color = :firebrick1
)

plot!(
    xvals,
    yfit,
    label="fit: c = $(round(c_best, digits=4))",
    linewidth=1,
    color = :black,
    linestyle = :dash
)

title!("Entanglement Entropy Scaling: PBC")