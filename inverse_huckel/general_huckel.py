import numpy as np

def calculate_distance(atom_i, atom_j): #calculates the Euclidean distance between two atoms based on their coordinates
    return np.linalg.norm(atom_i - atom_j) #atom i an atom j are numpy arrays representating the coordinates of two atoms

def construct_hamiltonian(coordinates, alpha, beta, cutoff_distance): #constructs hamiltonian of a molecules based on its atomic coordinates
    n = len(coordinates) #coordinates is a numpy array that contains the coordinates of all atoms in the molecule
    H = np.zeros((n, n))
    for i in range(n): #outer loop iterates over the indices of atoms in the molecule. i represents the index of the first atom.
        for j in range(i + 1, n): #inner loop iteraes over the indices of atoms adjacent to the atom i. i+1 to avoid redundant calcs and ensure that each pair of atoms is considered only once. j represents the index of the second atom.
            distance_ij = calculate_distance(coordinates[i], coordinates[j]) #calling the function to calculate distance between atoms i and j from the coordinates array.coordinates[i] is passed as the atom_i parameter, and coordinates[j] is passed as the atom_j parameter to the calculate_distance function.
            if distance_ij < cutoff_distance: #if distance between atoms i and j cooordinates are less than the cutoff distance
                H[i, j] = beta #off diagonal elements sets to beta
                H[j, i] = beta
    np.fill_diagonal(H, alpha) #diagonal elements filled with alpha
    return H #constructed hamiltonian is returned

# Example use of code:
benzene_coordinates = np.array([
    [-4.461121, 1.187057, -0.028519],
    [-3.066650, 1.263428, -0.002700],
    [-2.303848, 0.094131, 0.041626],
    [-2.935547, -1.151550, 0.059845],
    [-4.330048, -1.227982, 0.034073],
    [-5.092743, -0.058655, -0.010193]
])

alpha = -10  # Example value for alpha
beta = -1    # Example value for beta
cutoff_distance = 2.0  # Example cutoff distance

H_benzene = construct_hamiltonian(benzene_coordinates, alpha, beta, cutoff_distance)
print("Hamiltonian matrix for benzene:")
print(H_benzene)

#Solve the eigenvalue problem
energies, wavefunctions = np.linalg.eigh(H_benzene)

# Print the results
print("Eigenvalues (Energy levels):")
print(energies)
print("\nMolecular Orbitals (Eigenvectors):")
print(wavefunctions)


# def construct_hamiltonian(n, beta_values): #constructs hamiltonian matrix for cyclic pi system with n carbon atoms and a list of beta parameters
#     H = np.zeros((n, n)) #output is hamiltonian matrix representing the system
#     alpha = 0  # Energy of isolated atomic orbitals (set to 0 for simplicity)
#     for i in range(n):
#         H[i, i] = alpha
#         for j in range(1, len(beta_values) + 1): #complicated code
#             H[(i+j) % n, i] = beta_values[j-1]
#             H[i, (i+j) % n] = beta_values[j-1]
#     return H

# def solve_huckel_model(H): #solves the eigenvalue problem to obtain the molecular orbital energies
#     energies, wavefunctions = np.linalg.eigh(H)
#     return energies, wavefunctions

# Example usage
# n_benzene = 6
# beta_benzene = -1.0
# H_benzene = construct_hamiltonian(n_benzene, [beta_benzene])
# energies_benzene, _ = solve_huckel_model(H_benzene)
# print("Molecular orbital energies for benzene:", energies_benzene)

# n_naphthalene = 10
# beta_naphthalene = -0.5
# H_naphthalene = construct_hamiltonian(n_naphthalene, [beta_naphthalene])
# energies_naphthalene, _ = solve_huckel_model(H_naphthalene)
# print("Molecular orbital energies for naphthalene:", energies_naphthalene)

import matplotlib.pyplot as plt

# Example data (replace with your eigenvalues and eigenvectors)
energies = np.array([-10, -8, -6, -4])  # Eigenvalues (energies)
orbitals = np.array([[0.1, 0.2, 0.3, 0.4],  # Eigenvectors (molecular orbitals)
                     [0.2, 0.3, 0.4, 0.5],
                     [0.3, 0.4, 0.5, 0.6],
                     [0.4, 0.5, 0.6, 0.7]])

# Create figure and axes
fig, ax = plt.subplots()

# Plot molecular orbitals
for i, orbital in enumerate(orbitals):
    ax.plot(orbital, label=f'MO {i+1}')

# Plot energy levels
for energy in energies:
    ax.axhline(y=energy, color='gray', linestyle='--', linewidth=0.5)

# Add labels and legend
ax.set_xlabel('Atomic Index')
ax.set_ylabel('Energy (eV)')
ax.set_title('Molecular Orbitals and Energy Levels')
ax.legend()

# Show plot
plt.show()