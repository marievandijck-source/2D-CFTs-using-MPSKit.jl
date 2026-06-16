# Ising CFT – MPSKit.jl

Code for studying the critical transverse-field Ising chain as a CFT (c = 1/2),
written as part of a thesis at Ghent University. The main library is [MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl).

## Physics

At its critical point (J = g = 1) the transverse-field Ising chain is described in the
continuum limit by a free Majorana fermion CFT with central charge c = 1/2. The primary
operators are the identity 𝟙 (Δ = 0), the spin field σ (Δ = 1/8), and the energy density
ε (Δ = 1). This code extracts these quantities directly from finite-size numerics.

## Scripts

|File|Description|
|-|-|
|`Bulk_Free_Energy_and\Velocity.jl`|Bulk free energy density and velocity via finite-size extrapolation|
|`CFT_Spectrum.jl`|Energy spectrum → conformal tower and scaling dimensions|
|`Central_Charge.jl`|Central charge from entanglement entropy scaling (PBC)|
|`Energy_Spectrum.jl`|Raw energy spectrum via exact diagonalization|
|`OPE.jl`|OPE coefficient C\_σσε from three-point correlator|

Unlike the Heisenberg scripts, the Ising scripts are self-contained and do not share a
`Functions.jl` file.

## Method

**Bulk free energy density** (`ising\_bulk.jl`): the ground-state energy density E₀(L)/L
is extrapolated to L → ∞ using the finite-size ansatz

```
E(L)/L = e∞ + (πv/6) · (1/L²)
```

Exact values: e∞ = −4/π ≈ −1.2732, v = 2.

**CFT spectrum** (`ising\_cft\_spectrum.jl`): scaling dimensions are extracted via

```
Δ = (E − E₀) / (2πv/L),   v = 2
```

Momenta are assigned by diagonalizing the one-site translation operator within each
degenerate subspace. Expected primaries: Δ = 0, 1/8, 1.

**Central charge** (`ising\_central\_charge.jl`): the half-chain entanglement entropy is fit
to the Cardy–Calabrese formula (PBC):

```
S(ℓ) = (c/3) log\[(L/π) sin(πℓ/L)] + const
```

The convergence of c with bond dimension D is also plotted. Exact value: c = 1/2.

**Energy spectrum** (`ising\_energy\_spectrum.jl`): a quick exact-diagonalization check of
the 18 lowest energy levels.

**OPE coefficient** (`ising\_ope.jl`): the structure constant C\_σσε is estimated from the
ground-state four-point function via

```
C\_σσε ≈ ⟨Z\_i Z\_j Z\_k Z\_{k+1}⟩ / sqrt(⟨Z\_i Z\_j⟩² · ⟨Z\_k Z\_{k+1}⟩)
```

Exact value: C\_σσε = 1/2.

## Dependencies

* [MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl)
* [MPSKitModels.jl](https://github.com/QuantumKitHub/MPSKitModels.jl)
* [TensorKit.jl](https://github.com/Jutho/TensorKit.jl)
* [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl)
* Plots.jl

## Usage

Each script is self-contained. Set `L`, `D` (and `Ds` where applicable) at the top, then run:

```julia
include("ising\_cft\_spectrum.jl")
```


include("ising_cft_spectrum.jl")
```
