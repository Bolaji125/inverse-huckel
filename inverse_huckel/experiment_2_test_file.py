import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import torch.nn.functional as F

# Import necessary classes and functions from your plot and optimization script
from plot_alpha_beta_test_file import MolecularSystem
from plot_alpha_beta_test_file import optimise_molecular_system, plot_parameter_changes, plot_molecule, optimise_and_plot

# Define the coordinates of naphthalene
naphthalene_coordinates = np.array([
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
])

alpha_initial = -1.0
beta_initial = -1.0
cutoff_distance = 2.0

# Target eigenvalues for optimisation
target_eigenvalues_naphthalene = torch.tensor([-4.3, -3.6, -3.3, -3, -2.5, -1.2, -1.0, 1.3, 1.6, 2.3], dtype=torch.float32, requires_grad=False)

# Initialise the molecular system with naphthalene coordinates and initial alpha/beta values
molecular_system_naphthalene = MolecularSystem(naphthalene_coordinates, alpha_initial, beta_initial, cutoff_distance)

# Solve the eigenvalue problem to get the energies
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

# Plot the energy levels to visualise the HOMO-LUMO gap
molecular_system_naphthalene.plot_energy_levels_pytorch(energies_naphthalene, molecule_name="Naphthalene")

alpha_history, beta_history, loss_history = optimise_molecular_system(molecular_system_naphthalene, target_eigenvalues_naphthalene)
# Plot before and after optimisation with updated scaling factors
plot_molecule(molecular_system_naphthalene.coordinates, alpha_history[0], beta_history[0],
                  "Naphthalene - Before Optimisation", cutoff_distance, alpha_scale=500, beta_scale=15)
plt.savefig("before_optimisation_experiment_2.png")
    
plot_molecule(molecular_system_naphthalene.coordinates, alpha_history[-1], beta_history[-1],
                  "Naphthalene - After Optimisation", cutoff_distance, alpha_scale=500, beta_scale=15)
plt.savefig("after_optimisation_experiment_2.png")
 

# Recalculate the HOMO-LUMO gap after optimisation
optimised_energies_naphthalene = molecular_system_naphthalene.solve_eigenvalue_problem_pytorch()[0]

# Calculate the HOMO-LUMO gap for optimised energies
homo_optimised = optimised_energies_naphthalene[4]  # 5th element
lumo_optimised = optimised_energies_naphthalene[5]  # 6th element
homo_lumo_gap_optimised = lumo_optimised - homo_optimised

# Print the optimised HOMO-LUMO gap
print(f"The optimised HOMO-LUMO gap is {lumo_optimised:.6f} - {homo_optimised:.6f} = {homo_lumo_gap_optimised:.6f}")

plot_parameter_changes(alpha_history, beta_history, loss_history,"Naphthalene" , molecular_system_naphthalene.beta_indices)
plt.savefig("parameter_changes_experiment_2.png")





