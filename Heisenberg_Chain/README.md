# Heisenberg Model – CFT Spectrum via MPSKit.jl

Code for studying the spin-1/2 Heisenberg chain as a compactified free boson CFT (c = 1),
written as part of a thesis at Ghent University. The main library is [MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl).

## Physics

The antiferromagnetic Heisenberg chain is a critical model described in the continuum limit
by a compactified free boson CFT with central charge c = 1. This code extracts CFT data
directly from finite-size spectra and entanglement entropy scaling.

## Scripts

| File | Description |
|------|-------------|
| `Functions.jl` | Shared helper functions (see below) |
| `Bulk_Free_Energy_and_Velocity.jl` | Bulk free energy density and central charge via finite-size extrapolation |
| `CFT_Spectrum.jl` | Energy spectrum → conformal tower and scaling dimensions |
| `Central_Charge_OBC.jl` | Central charge from entanglement entropy scaling (open boundary conditions) |
| `Central_Charge_PBC.jl` | Central charge from entanglement entropy scaling (periodic boundary conditions) |

## Functions.jl

| Function | Description |
|----------|-------------|
| `heisenberg_hamiltonian(; J)` | SU(2)-symmetric Heisenberg MPO Hamiltonian |
| `bondspace_heis(D; jmax)` | SU(2) bond space with Gaussian multiplicity distribution |
| `ED_H(H, n)` | Exact diagonalization across SU(2) sectors |
| `DMRG_H(H, L, D, n)` | DMRG ground state + excitations across SU(2) sectors |
| `O_shift(L)` | One-site translation MPO on a ring |
| `fix_degeneracies(basis)` | Diagonalize translation operator within a degenerate subspace |
| `compute_momenta(energies, states)` | Assign momenta by diagonalizing translation within degenerate groups |
| `estimate_v(S, sectors, energies, L)` | Estimate spinon velocity from the finite-size spectrum |

`compute_momenta_mps` is an alias for `compute_momenta`.

## Method

**Bulk free energy density** (`heisenberg_bulk.jl`): the ground-state energy density E₀(L)/L
is computed for a range of system sizes and extrapolated to L → ∞ using the ansatz

```
E  (L) = e∞ · L + b/L
```

The central charge follows from the subleading coefficient: c = −6b/π.

**CFT spectrum** (`cft_spectrum.jl`): scaling dimensions are extracted from the finite-size
energy spectrum via

```
Δ = (E - E₀) / (2πv/L)
```

with velocity v = π/2 (exact for the Heisenberg chain). Momenta are assigned by
diagonalizing the translation operator within degenerate subspaces.

**Central charge from entanglement entropy**: the half-chain entanglement entropy is fit to
the Cardy–Calabrese formula,

```
S(ℓ) = (c/3) log[(L/π) sin(πℓ/L)] + const      (PBC)
S(ℓ) = (c/6) log[(2L/π) sin(πℓ/L)] + const     (OBC)
```

## Dependencies

- [MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl)
- [MPSKitModels.jl](https://github.com/QuantumKitHub/MPSKitModels.jl)
- [TensorKit.jl](https://github.com/Jutho/TensorKit.jl)
- [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl)
- Plots.jl
- JLD2.jl

## Usage

Set `L`, `D`, and `n` at the top of each script, then run e.g.:

```julia
include("Functions.jl")
include("cft_spectrum.jl")
```
