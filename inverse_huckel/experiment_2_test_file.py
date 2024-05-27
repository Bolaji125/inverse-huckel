import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import torch.nn.functional as F

# Import necessary classes and functions from your plot and optimization script
#from general_huckel_functionality import MolecularSystem
from plot_alpha_beta_test_file import MolecularSystem, \
    optimise_molecular_system, generate_atom_pairs, plot_parameter_changes, plot_molecule, optimise_and_plot
#from plot_alpha_beta_test_file import optimise_molecular_system, plot_parameter_changes, plot_molecule, optimise_and_plot

# Define the coordinates of naphthalene
naphthalene_coordinates = [
    [1.24593, 1.40391, -0.0000],
    [0.00001, 0.71731, -0.00000],
    [-0.00000, -0.71730, -0.00000],
    [1.24592, -1.40388, -0.00000],
    [2.43659, -0.70922, -0.00000],
    [2.43659, 0.70921, 0.00000],
    [-1.24593, -1.40387, 0.00000],
    [-2.43660, -0.70921, 0.00000],
    [-2.43660, 0.70921, 0.00000],
    [-1.24592, 1.40390, -0.00000],
]

alpha_initial = -1.0
beta_initial = -1.0
cutoff_distance = 1.5

# Target eigenvalues for optimisation
target_eigenvalues_naphthalene = torch.tensor([-4.3, -3.6, -3.3, -3, -2.5, -1.2, -1.0, 1.3, 1.6, 2.3], dtype=torch.float32, requires_grad=False)

# Initialize the molecular system with naphthalene coordinates and initial alpha/beta values
molecular_system_naphthalene = MolecularSystem(naphthalene_coordinates, alpha_initial, beta_initial, cutoff_distance)



# Solve the eigenvalue problem to get the energies
#energies_naphthalene = molecular_system_naphthalene.solve_eigenvalue_problem()[0]
energies_naphthalene = molecular_system_naphthalene.solve_eigenvalue_problem_pytorch()[0]

# Convert energies to regular decimal numbers with fixed decimal places (6 in this case)
formatted_energies = [f"{energy:.6f}" for energy in energies_naphthalene]
print("Energies naphthalene:", formatted_energies)

# Calculate the HOMO-LUMO gap
homo = energies_naphthalene[4]  # 5th element
lumo = energies_naphthalene[5]  # 6th element
homo_lumo_gap = lumo - homo

# Print the HOMO-LUMO gap
print(f"The HOMO-LUMO gap is {lumo:.6f} - {homo:.6f} = {homo_lumo_gap:.6f}")

# Plot the energy levels to visualize the HOMO-LUMO gap
molecular_system_naphthalene.plot_energy_levels_pytorch(energies_naphthalene, molecule_name="Naphthalene")

# Optimise alpha and beta parameters and plot the results
optimise_and_plot(molecular_system_naphthalene, target_eigenvalues_naphthalene, "Naphthalene", cutoff_distance)

# Recalculate the HOMO-LUMO gap after optimisation
optimised_energies_naphthalene = molecular_system_naphthalene.solve_eigenvalue_problem_pytorch()[0]

# Calculate the HOMO-LUMO gap for optimized energies
homo_optimised = optimised_energies_naphthalene[4]  # 5th element
lumo_optimised = optimised_energies_naphthalene[5]  # 6th element
homo_lumo_gap_optimised = lumo_optimised - homo_optimised

# Print the optimised HOMO-LUMO gap
print(f"The optimised HOMO-LUMO gap is {lumo_optimised:.6f} - {homo_optimised:.6f} = {homo_lumo_gap_optimised:.6f}")
