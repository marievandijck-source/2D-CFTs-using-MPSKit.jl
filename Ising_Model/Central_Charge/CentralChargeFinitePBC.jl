using MPSKit, MPSKitModels, TensorKit, Plots
using Logging
global_logger(NullLogger())

# Parameters 
L  = 10
Ds = 6:4:10
fitcuts = L÷3 : 2L÷3

# Build critical Ising Hamiltonian PBC
model = transverse_field_ising(Float64, Z2Irrep; J=1.0, g=1.0) 
H = periodic_boundary_conditions(model, L)

# Physical space 
phys = Z2Space(0 => 1, 1 => 1)

# Central charge estimation via entanglement entropy
function estimate_c(ψ; L::Int, fitcuts=fitcuts)
    # Von Neumann entropy at each cut ℓ = 1..L-1
    S = [entropy(ψ, ℓ) for ℓ in 1:(L-1)]

    # Entropy for PBC
    x = [log((L/π) * sin(π * ℓ / L)) for ℓ in 1:(L-1)]

    # linear least squares: y ≈ a + b x
    A = hcat(ones(length(fitcuts)), x[fitcuts])
    coeff = A \ S[fitcuts]
    intercept, slope = coeff[1], coeff[2]

    
    return 3*slope, slope, intercept, x, S # c = 3 * slope for PBC
end

# DMRG over bond dimensions
fits = Any[]
for D in Ds
    bond = Z2Space(0 => D ÷ 2, 1 => D ÷ 2)
    # Initial MPS
    ψ0 = FiniteMPS(L, phys, bond)
    # Ground state
    ψ, _, _ = find_groundstate(ψ0, H, DMRG())

    c, slope, intercept, x, S = estimate_c(ψ; L=L, fitcuts=fitcuts)
    push!(fits, (D=D, c=c, slope=slope, intercept=intercept, x=x, S=S))

    println("D = $D  ->  c ≈ $(round(c, digits=6))")
end

# Plot 1: entanglement entropy for largest D
best = fits[end]
xfit = range(minimum(best.x[fitcuts]), maximum(best.x[fitcuts]), length=200)

p1 = scatter(best.x, best.S;
    label="DMRG (D=$(best.D))",
    xlabel="ln((L/π) sin(πℓ/L))",
    ylabel="S(ℓ)",
    color=:maroon,
    legend=:topleft,
    dpi=300)

xfit = range(minimum(best.x), maximum(best.x), length=200)
plot!(p1, xfit, best.intercept .+ best.slope .* xfit;
    label="fit: c = $(round(c_best, digits=4)),\n DMRG D = $bestD, L = $L",
    linewidth=1,
    linestyle=:dash,
    color=:black)

title!(p1, "Entanglement Entropy Scaling")

# Plot 2: c vs D
D_vals = [f.D for f in fits]
c_vals = [f.c for f in fits]
p2 = scatter(D_vals, c_vals;
    label="c(D)",
    xlabel="Bond dimension D",
    ylabel="c",
    color=:maroon,
    legend=:bottomright,
    dpi=300)

hline!(p2, [0.5];
    label="Exact c = 1/2",
    linestyle=:dash,
    color=:black)

title!(p2, "Central Charge vs Bond Dimension")

display(p1)
display(p2)

savefig(p1, "Ising_c_fit.png")
savefig(p2, "Ising_c_vs_D.png")