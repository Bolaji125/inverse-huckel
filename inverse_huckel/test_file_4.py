import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
from matplotlib import cm

class MolecularSystem:

    def __init__(self, coordinates, alpha, beta, cutoff_distance):
        self.coordinates = torch.tensor(coordinates, dtype=torch.float32)
        self.alpha = torch.tensor([alpha] * len(coordinates), requires_grad=True)
        self.beta = torch.tensor([beta] * len(coordinates), requires_grad=True)  # Modified to accept beta as a tensor
        self.cutoff_distance = cutoff_distance
        self.H = torch.zeros((len(coordinates), len(coordinates)), dtype=torch.float32)
        self.update_hamiltonian()

    def calculate_distance(self, atom_i, atom_j):
        atom_i_tensor = torch.tensor(atom_i)
        atom_j_tensor = torch.tensor(atom_j)
        return torch.linalg.norm(atom_i_tensor - atom_j_tensor)

    def update_hamiltonian(self):
        num_atoms = self.coordinates.shape[0]
        self.H = torch.zeros((num_atoms, num_atoms), dtype=torch.float32)
        
        # Set diagonal elements to alpha values
        for i in range(num_atoms):
            self.H[i, i] = self.alpha[i]
        
        # Set off-diagonal elements based on distance and beta
        for i in range(num_atoms):
            #for j in range(i + 1, num_atoms):
            for j in range(num_atoms):  # Iterate over all atoms
                if i != j:  # Skip diagonal elements, as they are alpha parameters
                    distance = torch.norm(self.coordinates[i] - self.coordinates[j])
                    if distance <= self.cutoff_distance:
                        self.H[i, j] = self.beta[i]  # Modified to use beta values
                        self.H[j, i] = self.beta[i]

    def solve_eigenvalue_problem_pytorch(self):
        eigenvalues, eigenvectors = torch.linalg.eigh(self.H)
        return eigenvalues, eigenvectors

# Optimization loop with target alpha and beta parameters
alpha_initial = -10.0
beta_initial = -1.0
cutoff_distance = 2.0

benzene_coordinates = np.array([
    [-4.461121, 1.187057, -0.028519],
    [-3.066650, 1.263428, -0.002700],
    [-2.303848, 0.094131, 0.041626],
    [-2.935547, -1.151550, 0.059845],
    [-4.330048, -1.227982, 0.034073],
    [-5.092743, -0.058655, -0.010193]
])

# Define the target alpha and beta parameters
target_alpha_parameters = torch.tensor([-9.0, -9.1, -9.2, -9.3, -9.4, -9.5], requires_grad=False)
target_beta_parameters = torch.tensor([-0.9, -0.95, -1.0, -1.05, -1.1, -1.15], requires_grad=False)

# Create molecular system instance
molecular_system_benzene = MolecularSystem(benzene_coordinates, alpha_initial, beta_initial, cutoff_distance)

# Set initial alpha and beta values
molecular_system_benzene.alpha = torch.tensor([-10.0, -10.0, -10.0, -10.0, -10.0, -10.0], requires_grad=True)
molecular_system_benzene.beta = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0], requires_grad=True)

# Learning rate
learning_rate = 0.01
eigenvalues, _ = molecular_system_benzene.solve_eigenvalue_problem_pytorch()
print("eigenvalues before optimisation:", eigenvalues)
print("hamiltonian:",molecular_system_benzene.H )
# Optimization loop
for i in range(2000):
    molecular_system_benzene.update_hamiltonian()
    eigenvalues, _ = molecular_system_benzene.solve_eigenvalue_problem_pytorch()

    # Calculate loss based on the difference between current alpha and beta parameters and target parameters
    loss_alpha = F.mse_loss(molecular_system_benzene.alpha, target_alpha_parameters)
    loss_beta = F.mse_loss(molecular_system_benzene.beta, target_beta_parameters)
    loss = loss_alpha + loss_beta

    loss.backward()

    with torch.no_grad():
        molecular_system_benzene.alpha -= learning_rate * molecular_system_benzene.alpha.grad
        molecular_system_benzene.beta -= learning_rate * molecular_system_benzene.beta.grad

    molecular_system_benzene.alpha.grad.zero_()
    molecular_system_benzene.beta.grad.zero_()

# Print the optimized alpha and beta values and resulting Hamiltonian and eigenvalues
print("Optimized Alpha Values:", molecular_system_benzene.alpha)
print("Optimized Beta Values:", molecular_system_benzene.beta)
print("Optimized Hamiltonian:", molecular_system_benzene.H)
eigenvalues_after, _ = molecular_system_benzene.solve_eigenvalue_problem_pytorch()
print("Eigenvalues after optimization:", eigenvalues_after.real)