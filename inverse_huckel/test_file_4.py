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
            for j in range(num_atoms):  # Iterate over all atoms
                if i != j:  # Skip diagonal elements, as they are alpha parameters
                    distance = torch.norm(self.coordinates[i] - self.coordinates[j])
                    if distance <= self.cutoff_distance:
                        self.H[i, j] = self.beta[i]  # Modified to use beta values
                        self.H[j, i] = self.beta[i]

    def solve_eigenvalue_problem_pytorch(self):
        eigenvalues, eigenvectors = torch.linalg.eigh(self.H)
        return eigenvalues, eigenvectors

# Function for optimization loop
def optimize_molecular_system(molecular_system, target_eigenvalues, num_iterations=1000, learning_rate=0.01):
    # Learning rate
    eigenvalues, _ = molecular_system.solve_eigenvalue_problem_pytorch()
    print("eigenvalues before optimization:", eigenvalues)
    print("hamiltonian:", molecular_system.H)

    # Optimization loop
    for i in range(num_iterations + 1):
        molecular_system.update_hamiltonian()
        eigenvalues, _ = molecular_system.solve_eigenvalue_problem_pytorch()

        # Calculate loss based on the difference between current and target eigenvalues
        loss = F.mse_loss(eigenvalues, target_eigenvalues)
        if i % 500 == 0 or i == num_iterations:  # Print every 100 iterations
            print(f"Iteration {i}, Loss: {loss.item()}")

        if i < num_iterations:  # Only perform gradient descent for the first num_iterations iterations
            loss.backward()



        with torch.no_grad():
            molecular_system.alpha -= learning_rate * molecular_system.alpha.grad
            molecular_system.beta -= learning_rate * molecular_system.beta.grad

        molecular_system.alpha.grad.zero_()
        molecular_system.beta.grad.zero_()

    # Print the optimized alpha and beta values and resulting Hamiltonian and eigenvalues
    print("Optimized Alpha Values:", molecular_system.alpha)
    print("Optimized Beta Values:", molecular_system.beta)
    print("Optimized Hamiltonian:", molecular_system.H)
    eigenvalues_after, _ = molecular_system.solve_eigenvalue_problem_pytorch()
    print("Eigenvalues after optimization:", eigenvalues_after.real)

# Constants
alpha_initial = -10.0
beta_initial = -1.0
cutoff_distance = 2.0

# Define target eigenvalues for benzene
target_eigenvalues_benzene = torch.tensor([-13.0, -12.0, -10.0, -10.0, -10.0, -9.0], dtype=torch.float32, requires_grad=False)

# Define target eigenvalues for napthalene
target_eigenvalues_napthalene = torch.tensor([-13.0, -12.0, -11.5, -12.5, -11.0, -10.5, -10.0, -9.0, -9.5, -8.0], dtype=torch.float32, requires_grad=False)  # Add your target eigenvalues for napthalene here

# Benzene coordinates
benzene_coordinates = np.array([
    [-4.461121, 1.187057, -0.028519],
    [-3.066650, 1.263428, -0.002700],
    [-2.303848, 0.094131, 0.041626],
    [-2.935547, -1.151550, 0.059845],
    [-4.330048, -1.227982, 0.034073],
    [-5.092743, -0.058655, -0.010193]
])

# Napthalene coordinates
napthalene_coordinates = np.array([
    [ 1.24593,1.40391, -0.0000],
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

# Create molecular system instances
molecular_system_benzene = MolecularSystem(benzene_coordinates, alpha_initial, beta_initial, cutoff_distance)
molecular_system_napthalene = MolecularSystem(napthalene_coordinates, alpha_initial, beta_initial, cutoff_distance)

# Optimize molecular systems
optimize_molecular_system(molecular_system_benzene, target_eigenvalues_benzene)
optimize_molecular_system(molecular_system_napthalene, target_eigenvalues_napthalene)






