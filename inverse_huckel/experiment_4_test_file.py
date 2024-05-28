import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import torch.nn.functional as F

# Import necessary classes and functions from your plot and optimization script
from plot_alpha_beta_test_file import MolecularSystem
from plot_alpha_beta_test_file import optimise_molecular_system, plot_parameter_changes, plot_molecule, optimise_and_plot

alpha_initial = -1.0
beta_initial = -1.0
benzene_cutoff_distance = 1.0

target_eigenvalues_benzene = torch.tensor([-2.0, -1.0, -1.0, 1.0, 1.0, 2.0], dtype=torch.float32, requires_grad=False)

benzene_coordinates = np.array([
    [1.0, 0.0, 0.0],
    [0.5, np.sqrt(3)/2, 0.0],
    [-0.5, np.sqrt(3)/2, 0.0],
    [-1.0, 0.0, 0.0],
    [-0.5, -np.sqrt(3)/2, 0.0],
    [0.5, -np.sqrt(3)/2, 0.0]
])

# Define learning rates to investigate
learning_rates = [0.1, 0.01, 0.05]

# Initialize lists to store results
alpha_histories = []
beta_histories = []
loss_histories = []

# Iterate over different learning rates
for lr in learning_rates:
    # Create instance for benzene
    molecular_system_benzene = MolecularSystem(benzene_coordinates, alpha_initial, beta_initial, benzene_cutoff_distance)
    # Perform optimization and store the histories
    alpha_history, beta_history, loss_history = optimise_molecular_system(molecular_system_benzene, target_eigenvalues_benzene, learning_rate=lr)
    alpha_histories.append(alpha_history)
    beta_histories.append(beta_history)
    loss_histories.append(loss_history)

# # Plot parameter changes against the number of epochs for each learning rate
# for i, lr in enumerate(learning_rates):
#     plot_parameter_changes(alpha_histories[i], beta_histories[i], loss_histories[i], f"Benzene (LR={lr})", molecular_system_benzene.beta_indices)
# plt.show()
# Plot parameter changes against the number of epochs for each learning rate
for i, lr in enumerate(learning_rates):
    plt.figure()  # Create a new figure for each plot
    plot_parameter_changes(alpha_histories[i], beta_histories[i], loss_histories[i], f"Benzene (LR={lr})", molecular_system_benzene.beta_indices)
    plt.savefig(f"parameter_changes_LR_{lr}.png")  # Save the figure as a PNG file
    plt.show()  # Display the figure