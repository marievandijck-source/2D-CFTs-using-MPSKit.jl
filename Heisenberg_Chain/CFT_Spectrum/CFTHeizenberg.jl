# Save the outputs (v, energies, momenta, L, D, figure)
using MPSKit, MPSKitModels, TensorKit, KrylovKit
using LinearAlgebra, Printf, Statistics, Plots, JLD2, CSV, DataFrames

include("Functions.jl")

LinearAlgebra.BLAS.set_num_threads(8)

# Parameters 
#=
L = parse(Int, ARGS[1])
D = parse(Int, ARGS[2])
n = parse.(Int, split(ARGS[3], "_"))
calculation = ARGS[4]
=#
# Parameters 
L = 48
D = 100
n = [5,5,1]
calculation = "DMRG"

t = time()


# Hamiltonian
H = periodic_boundary_conditions(heisenberg_hamiltonian(), L)

# Energy spectrum (Singlets and triplets)
if calculation == "ED"
   energies, states, sectors = ED_H(H, n)
else
    energies, states, sectors = DMRG_H(H, L, D, n)
end

# Conformal momenta
momenta = compute_momenta_mps(energies, states)
S = momenta ./ (2π / L)

# Estimation of v
#v = estimate_v(S)
#println("schatting van v = $v")  
v = π/2 

# Scaling dimensions
Δ = real.(energies[1:length(energies)] .- energies[1]) ./ (2π * v / L) 


# Plot
p = plot(
    S, real.(Δ);
    seriestype = :scatter,
    legend = false,
    label = "",
    markerstrokewidth = 0.5,
    framestyle = :box,
    grid = :y, 
    gridalpha = 0.2,
    dpi=300
)

m = ceil(Int, L/10)

max_y = ceil(Int, maximum(Δ))

int = Float64[]
append!(int, 0)
half = Float64[]
for i in 1:max_y
    append!(int, i)
    append!(half, i - 0.5)
end

hline!(p, half; color = "lightgrey", label="")
hline!(p, int; color = "lightgrey", label="")

# Mask
mask_pi = isapprox.(abs.(S), L/2; atol=1e-4)
mask_two = isapprox.(abs.(S), 2.0; atol=1e-4)
mask_S0 = (sectors .== 0)
mask_S1 = (sectors .== 1)

mask_T_basis = (sectors .== 0) .& isapprox.(abs.(S), 2.0; atol=1e-4)
mask_J_basis = (sectors .== 1) .& isapprox.(abs.(S), 1.0; atol=1e-4)

idx_T = findall(mask_T_basis)
idx_J = findall(mask_J_basis)

sorted_idx_T = idx_T[sortperm(real.(Δ[idx_T]))]
sorted_idx_J = idx_J[sortperm(real.(Δ[idx_J]))]

final_idx_T = sorted_idx_T[1:min(2, end)]
final_idx_J = sorted_idx_J[1:min(2, end)]

mask_T = falses(length(S))
mask_T[final_idx_T] .= true

mask_J = falses(length(S))
mask_J[final_idx_J] .= true


sector_colors = [:maroon, :firebrick1, :navajowhite, :indianred]
sector_labels = ["Singlets (S=0)", "Triplets (S=1)", "Quintets (S=2)", "Septets (S=7/2)"]


for s_idx in 0:(length(n) - 1)
    mask_sector = (sectors .== s_idx)
    
    col = s_idx + 1 <= length(sector_colors) ? sector_colors[s_idx + 1] : :black
    lab = s_idx + 1 <= length(sector_labels) ? sector_labels[s_idx + 1] : "S=$s_idx"

    scatter!(p, S[mask_sector], real.(Δ[mask_sector]), 
             label = lab, color = col, markershape = :circle)


    mask_pi_sector = mask_sector .& isapprox.(abs.(S), L/2; atol=1e-2)
    
    if any(mask_pi_sector)
        # Plot de randpunten (we gebruiken primary=false zodat ze de legende niet vervuilen)
        scatter!(p, fill(-L/2, sum(mask_pi_sector)), real.(Δ[mask_pi_sector]), 
                 label = "", color = col, markershape = :circle)
        scatter!(p, fill(L/2, sum(mask_pi_sector)), real.(Δ[mask_pi_sector]), 
                 label = "", color = col, markershape = :circle)
    end
end

scatter!(p, S[mask_T], Δ[mask_T] , label = "Stress–energy tensor",  color = :maroon, markershape = :diamond, markersize = 7)
scatter!(p, S[mask_J], Δ[mask_J] , label = "Current",  color = :firebrick1, markershape = :diamond, markersize = 7)



plot!(p, 
    legend = :bottomleft, 
    legendtitlefontsize = 9,
    legendfontsize = 8
)

xlabel!(p, "Momentum S [units of 2π/L]")
ylabel!(p, "Scaling Dimension Δ")
hoofdtitel = "Conformal Spectrum - Heisenberg Chain"

title!(p, "$hoofdtitel",
titlefontsize = 12)

marge = L / 20 
x_lims = (-(L/2) - marge, (L/2) + marge)

stap = L / 4

plot!(p, 
    xlims = x_lims, 
    xticks = -(L/2):stap:(L/2)
)


savefig(p, "plot_L$(L)_D=$(D)_$(calculation).png")

jldsave("data_L$(L)_D=$(D)_$(calculation).jld2"; energies, Δ, S, L, sectors)


dt = time() - t
@info "Tijd: $dt seconden"

p