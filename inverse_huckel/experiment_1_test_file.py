import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os


from plot_alpha_beta_test_file import MolecularSystem, \
    optimise_molecular_system, generate_atom_pairs, plot_parameter_changes, plot_molecule, optimise_and_plot

alpha_initial = -1.0
beta_initial = -1.0
cutoff_distance = 1.0  # Adjusted cutoff distance for the perfect hexagon

# Target eigenvalues for benzene
target_eigenvalues_benzene = torch.tensor([-2.0, -0.5, -0.5, 0.5, 0.5, 2.0], dtype=torch.float32, requires_grad=False)

# Coordinates for benzene (perfect hexagon)
benzene_coordinates = np.array([
    [1.0, 0.0, 0.0],
    [0.5, np.sqrt(3)/2, 0.0],
    [-0.5, np.sqrt(3)/2, 0.0],
    [-1.0, 0.0, 0.0],
    [-0.5, -np.sqrt(3)/2, 0.0],
    [0.5, -np.sqrt(3)/2, 0.0]
])

molecule_name ="Benzene"

# Create an instance for benzene
molecular_system_benzene = MolecularSystem(benzene_coordinates, alpha_initial, beta_initial, cutoff_distance)

alpha_history, beta_history, loss_history = optimise_molecular_system(molecular_system_benzene, target_eigenvalues_benzene)
    
    # Plot before and after optimisation with updated scaling factors
plot_molecule(molecular_system_benzene.coordinates, alpha_history[0], beta_history[0],
                  f"{molecule_name} - Before Optimisation", cutoff_distance, alpha_scale=500, beta_scale=15)
plt.savefig("before_optimisation_experiment_1.png")
    
plot_molecule(molecular_system_benzene.coordinates, alpha_history[-1], beta_history[-1],
                  f"{molecule_name} - After Optimisation", cutoff_distance, alpha_scale=500, beta_scale=15)
plt.savefig("after_optimisation_experiment_1.png")
 
plot_parameter_changes(alpha_history, beta_history, loss_history, molecule_name, molecular_system_benzene.beta_indices)
try:
    # Save the figure
    plt.savefig("parameter_changes_experiment_1.png")
    print("Image saved successfully.")
except Exception as e:
    print("Error occurred while saving the image:", e)
#plt.savefig("parameter_changes_experiment_1.png")




