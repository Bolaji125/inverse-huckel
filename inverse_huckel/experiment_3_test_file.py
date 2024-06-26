import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import torch.nn.functional as F

# Import necessary classes and functions from your plot and optimization script
from plot_alpha_beta_test_file import MolecularSystem
from plot_alpha_beta_test_file import optimise_molecular_system, plot_parameter_changes, plot_molecule, optimise_and_plot

# Define the coordinates of benzene
# benzene_coordinates = np.array([
#     [1.0, 0.0, 0.0],
#     [0.5, np.sqrt(3)/2, 0.0],
#     [-0.5, np.sqrt(3)/2, 0.0],
#     [-1.0, 0.0, 0.0],
#     [-0.5, -np.sqrt(3)/2, 0.0],
#     [0.5, -np.sqrt(3)/2, 0.0]
# ])
benzene_coordinates = np.array([ # scaled benzene coordinates
    [ 1.3965    ,  0.        ,  0.        ],
    [ 0.69825   ,  1.2095694 ,  0.        ],
    [-0.69825   ,  1.2095694 ,  0.        ],
    [-1.3965    ,  0.        ,  0.        ],
    [-0.69825   , -1.2095694 ,  0.        ],
    [ 0.69825   , -1.2095694 ,  0.        ]
])

# Set parameters
alpha_initial_range = (-1.5, -0.8)
beta_initial_range = (-1.5, -0.8)
cutoff_distance = 2.0

# Generate random alpha and beta values within the specified range
random_alphas = np.random.uniform(*alpha_initial_range, size=(6,))
random_betas = np.random.uniform(*beta_initial_range, size=(6,))

for (alpha, beta) in zip(random_alphas, random_betas):

    # Print random alpha and beta values
    print(f"Random Alpha Values: {alpha}")
    print(f"Random Beta Values: {beta}")

    # Target eigenvalues for benzene
    target_eigenvalues_benzene = torch.tensor([-2.0, -1.0, -1.0, 1.0, 1.0, 2.0], dtype=torch.float32, requires_grad=False)

    # Create instance of MolecularSystem with random alpha and beta values
    molecular_system_benzene = MolecularSystem(benzene_coordinates, alpha, beta, cutoff_distance)

    # Perform optimization
    alpha_history, beta_history, loss_history = optimise_molecular_system(molecular_system_benzene, target_eigenvalues_benzene)

    # Plot parameter changes
    plot_parameter_changes(alpha_history, beta_history, loss_history, "Benzene", molecular_system_benzene.beta_indices)
    #plt.savefig("parameter_changes_experiment_3.png")

    # Plot molecule before and after optimisation
    plot_molecule(molecular_system_benzene.coordinates, alpha_history[0], beta_history[0],
                "Benzene - Before Optimisation", cutoff_distance)
    #plt.savefig("before_optimisation_experiment_3.png")
    plot_molecule(molecular_system_benzene.coordinates, alpha_history[-1], beta_history[-1],
                "Benzene - After Optimisation", cutoff_distance)
    #plt.savefig("after_optimisation_experiment_3.png")

   