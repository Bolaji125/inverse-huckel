import numpy as np
import matplotlib.pyplot as plt
import turtle

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


# #plot of energy levels-ASK ABOUT DEGENERACY
# x_values = np.zeros_like(energies)  # Use zeros for x-values (all points aligned)
# #plt.scatter(x_values, energies, color='blue', label='Energy Levels')
# #plt.plot([0]*len(energies), energies, linestyle='--', color='blue', label='Energy Levels')
# plt.scatter(x_values, energies, linestyle='-', color='blue', label='Energy Levels')

# # Add labels and title
# plt.xlabel('Index')
# plt.ylabel('Energy (eV)')
# plt.title('Energy Levels of Benzene')

# # Show plot
# plt.show()

# benzene using turtle

# Creating the window
# window = turtle.Screen()
# window.setup(600, 600, startx=0, starty=100)
# window.bgcolor('black')

# # Colours
# #colors = ['red', 'purple', 'blue', 'green', 'orange', 'yellow']

# # Creating the turtle
# t = turtle.Turtle()
# t.speed(0)  # Set the fastest speed
# t.color('cyan')

# #benzene using 1 colour
# for x in range(100): 
#     t.width(x // 10 + 1)
#     t.forward(x)
#     t.left(59)  # Adjust the angle to get the benzene shape

# # # Keeping the window open
# turtle.done()

#benzene using matplotlib

# Define the coordinates of the benzene molecule
benzene_coordinates = np.array([
    [0, 1],
    [np.sqrt(3)/2, 0.5], #The np.sqrt(3)/2 value is used to ensure that the bonds have equal length, which is important for the structure of the benzene molecule.
    [np.sqrt(3)/2, -0.5],
    [0, -1],
    [-np.sqrt(3)/2, -0.5],
    [-np.sqrt(3)/2, 0.5],
    [0, 1]  # Closing the loop
])

# Define wavefunction sizes and phases (example values)
wavefunction_sizes = [0.2, 0.5, 0.8, 1.0, 0.7, 0.4, 0.1]
wavefunction_phases = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan']

# Plot benzene coordinates
plt.figure(figsize=(6, 6))
plt.plot(benzene_coordinates[:, 0], benzene_coordinates[:, 1], 'o-', color='black')

#Plot circles with different sizes and colors
for i, (x, y) in enumerate(benzene_coordinates):
    size = wavefunction_sizes[i] * 100  # Adjust size for better visualization
    color = wavefunction_phases[i]
    circle = plt.Circle((x, y), size, color=color, alpha=0.5)
    plt.gca().add_artist(circle)


# Plot the benzene molecule
# plt.figure(figsize=(6, 6))
# plt.plot(benzene_coordinates[:, 0], benzene_coordinates[:, 1], 'o-', color='blue')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Benzene Molecule')
# plt.axis('equal')  # Equal aspect ratio
# plt.grid(True)
# plt.show()


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

