import numpy as np

def calculate_distance(atom_i, atom_j): #calculates the Euclidean distance between two atoms based on their coordinates
    return np.linalg.norm(atom_i - atom_j) #atom i an atom j are numpy arrays representating the coordinates of two atoms

def construct_hamiltonian(coordinates, alpha, beta, cutoff_distance): #constructs hamiltonian of a molecules based on its atomic coordinates
    n = len(coordinates) #coordinates is a numpy array that contains the coordinates of all atoms in the molecule
    H = np.zeros((n, n))
    for i in range(n): #function iterates over all pairs of atoms
        for j in range(i + 1, n):
            distance_ij = calculate_distance(coordinates[i], coordinates[j]) #calling the function to calculate distance between atoms i and j
            if distance_ij < cutoff_distance: #if distance between atoms i and j cooordinates are less than the cutoff distance
                H[i, j] = beta #off diagonal elements sets to beta
                H[j, i] = beta
    np.fill_diagonal(H, alpha) #diagonal elements filled with alpha
    return H #constructed hamiltonian is returned

# Example usage:
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


# def construct_hamiltonian(n, beta_values): #constructs hamiltonian matrix for cyclic pi system with n carbon atoms and a list of beta parameters
#     H = np.zeros((n, n)) #output is hamiltonian matrix representing the system
#     alpha = 0  # Energy of isolated atomic orbitals (set to 0 for simplicity)
#     for i in range(n):
#         H[i, i] = alpha
#         for j in range(1, len(beta_values) + 1):
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