import numpy as np
import matplotlib.pyplot as plt

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
# # Plot each individual molecular orbital on its own graph
# benzene_coordinates_graph = np.array([
#     [-4.461121, 1.187057, -0.028519],
#     [-3.066650, 1.263428, -0.002700],
#     [-2.303848, 0.094131, 0.041626],
#     [-2.935547, -1.151550, 0.059845],
#     [-4.330048, -1.227982, 0.034073],
#     [-5.092743, -0.058655, -0.010193],
#     [-4.461121, 1.187057, -0.028519] #to close the loop
# ])
# num_mos = len(wavefunctions[0]) #calculates length of mo by counting the length of the first row of wavefunctions array
# for i in range(num_mos): #iterates over each molecular orbital
#     plt.figure(figsize=(6, 6)) #creates a new figure for each molecular orbital
#     plt.plot(benzene_coordinates_graph[:, 0], benzene_coordinates_graph[:, 1], 'o-', color='blue') #plots benzene using its x and y coordinates. o says that markers should be used and lines should connect the markers
#     circle_sizes = np.abs(wavefunctions[:, i]) * 5000 #the absolute values of the mo from wavefunctions are multiplied to scale them for visualisation. wavefunctions syntax: selects the ith mo and all rows of the wavefunction array
#     colors = ['green' if val >= 0 else 'red' for val in wavefunctions[:, i]] #if coefficient is positive then circle green, if negative circle red
#     for j in range(len(benzene_coordinates)): #plot a scatter point for each atom in the benzene molecule.
#         plt.scatter(benzene_coordinates_graph[j, 0], benzene_coordinates_graph[j, 1], s=circle_sizes[j], color=colors[j], alpha=0.5)
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title(f'Molecular Orbital {i+1}') #line sets the title of the plot to indiciate the mo being plotted
#     plt.axis('equal') #ensures circles appear circular rather than stretched
#     plt.grid(True)
#     plt.show()


# #plot of energy levels-ASK ABOUT DEGENERACY
# #x_values = np.zeros_like(energies)  # Use zeros for x-values (all points aligned)
# x_values_benzene = [0, 0, 0.1, 0, 0.1, 0]
# y_values_benzene = [-12, -11, -11, -9, -9, -8]
# plt.scatter(x_values_benzene, y_values_benzene, linestyle='-', color='blue', label='Energy Levels')

# # Set the x-axis tick locations and labels
# #plt.xticks([0, 0.1], ['0', '0.1'])

# # Adjust the margins to make the x-axis tick marks visually closer
# plt.margins(x=5)

# # Plot the energy levels as horizontal lines
# for y in y_values_benzene:
#     plt.hlines(y, xmin=0, xmax=0.1, color='blue')

# # Add labels and title
# plt.xlabel('Index')
# plt.ylabel('Energy (eV)')
# plt.title('Energy Levels of Benzene')

# # Show plot
# plt.show()

# # Define the energy levels
# x_values_benzene = [0, 0, 0.1, 0, 0.1, 0]
# y_values_benzene = [-12, -11, -11, -9, -9, -8]

# # Plot the energy levels as horizontal lines
# for y in y_values_benzene:
#     plt.hlines(y, xmin=0, xmax=0.1, color='blue')

# # Set the x-axis tick locations and labels
# plt.xticks([0, 0.1], ['0', '0.1'])

# # Adjust the margins to make the x-axis tick marks visually closer
# plt.margins(x=5)

# # Add labels and title
# plt.xlabel('Index')
# plt.ylabel('Energy (eV)')
# plt.title('Energy Levels of Benzene')

# # Show plot
# plt.show()

# Define the energy levels
x_values_benzene = [0, 0, 0.1, 0, 0.1, 0]
y_values_benzene = [-12, -11, -11, -9, -9, -8]

# Plot the energy levels as horizontal lines
for i in range(len(x_values_benzene)): #this loop iterates over each energy level defined
    plt.hlines(y_values_benzene[i], xmin=x_values_benzene[i], xmax=x_values_benzene[i] + 0.08, color='blue') #adds a horizontal line on each coordinate

# Adjust the margins to make the x-axis tick marks visually closer
plt.margins(x=3)

# Add labels and title
#plt.xlabel('Index')
plt.ylabel('Energy (eV)')
plt.title('Energy Levels of Benzene')

# Show plot
plt.show()











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

