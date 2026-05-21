# Energy Spectrum

Here you find the basic code (energy_spectrum.jl) to find the energy spectrum 
of the 18 most low lying states of the critical transverse field Ising model. 
The code is based on the 
example in [MPSKit.jl](https://quantumkithub.github.io/MPSKit.jl/stable/examples/quantum1d/1.ising-cft/#The-Ising-CFT-spectrum). 

The code uses exact diagonalization an a matrix product operator (MPO) Hamiltonian constructed on a chain of L=12 spins with periodic boudary conditins (PBC). 
The results are shown in the plot (energy_spectrum.png).
