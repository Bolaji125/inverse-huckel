import numpy as np
import matplotlib.pyplot as plt

class MolecularSystem:
    def __init__(self, coordinates, alpha, beta, cutoff_distance):
        self.coordinates = coordinates
        self.alpha = alpha
        self.beta = beta
        self.cutoff_distance = cutoff_distance
        self.H = self.construct_hamiltonian()

    def calculate_distance(self, atom_i, atom_j):
        return np.linalg.norm(atom_i - atom_j)

    def construct_hamiltonian(self):
        n = len(self.coordinates)
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                distance_ij = self.calculate_distance(self.coordinates[i], self.coordinates[j])
                if distance_ij < self.cutoff_distance:
                    H[i, j] = self.beta
                    H[j, i] = self.beta
        np.fill_diagonal(H, self.alpha)
        return H

    def solve_eigenvalue_problem(self):
        return np.linalg.eigh(self.H)
    
    def plot_molecular_orbitals(self, coordinates): #doesn't work
        energies, wavefunctions = self.solve_eigenvalue_problem()
        molecule_coordinates_graph = np.append(coordinates, [coordinates[0]], axis=0)
        num_mos = len(wavefunctions[0])

        # Draw lines for pairs of atoms with short distances
        threshold_distance = 1.5  # Adjust this threshold as needed
        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):
                distance = np.linalg.norm(coordinates[i] - coordinates[j])
                if distance <= threshold_distance:
                    plt.plot([coordinates[i, 0], coordinates[j, 0]], [coordinates[i, 1], coordinates[j, 1]], 'k-')

        for i in range(num_mos):
            plt.figure(figsize=(6, 6))
            plt.plot(molecule_coordinates_graph[:, 0], molecule_coordinates_graph[:, 1], 'o-', color='blue')
            circle_sizes = np.abs(wavefunctions[:, i]) * 5000
            colors = ['green' if val >= 0 else 'red' for val in wavefunctions[:, i]]
            for j in range(len(coordinates)):
                plt.scatter(coordinates[j, 0], coordinates[j, 1], s=circle_sizes[j], color=colors[j], alpha=0.5)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Molecular Orbital {i+1}')
            plt.axis('equal')
            plt.grid(True)
            plt.show()
    
    
    

    # def plot_molecular_orbitals(self):
    #     energies, wavefunctions = self.solve_eigenvalue_problem()
    #     benzene_coordinates_graph = np.append(self.coordinates, [self.coordinates[0]], axis=0)
    #     num_mos = len(wavefunctions[0])
    #     for i in range(num_mos):
    #         plt.figure(figsize=(6, 6))
    #         plt.plot(benzene_coordinates_graph[:, 0], benzene_coordinates_graph[:, 1], 'o-', color='blue')
    #         circle_sizes = np.abs(wavefunctions[:, i]) * 5000
    #         colors = ['green' if val >= 0 else 'red' for val in wavefunctions[:, i]]
    #         for j in range(len(self.coordinates)):
    #             plt.scatter(self.coordinates[j, 0], self.coordinates[j, 1], s=circle_sizes[j], color=colors[j], alpha=0.5)
    #         plt.xlabel('X')
    #         plt.ylabel('Y')
    #         plt.title(f'Molecular Orbital {i+1}')
    #         plt.axis('equal')
    #         plt.grid(True)
    #         plt.show()
    # def plot_molecular_orbitals(self, coordinates):
    #     print("coordinates:", coordinates)
    #     energies, wavefunctions = self.solve_eigenvalue_problem()
    #     print("energies:", energies)
    #     print("wavefunctions:", wavefunctions)
    #     molecule_coordinates_graph = np.append(self.coordinates, [self.coordinates[0]], axis=0)
    #     num_mos = len(wavefunctions[0])
    #     for i in range(num_mos):
    #         plt.figure(figsize=(6, 6))
    #         plt.plot(molecule_coordinates_graph[:, 0], molecule_coordinates_graph[:, 1], 'o-', color='blue')
    #         circle_sizes = np.abs(wavefunctions[:, i]) * 5000
    #         print(circle_sizes)
    #         colors = ['green' if val >= 0 else 'red' for val in wavefunctions[:, i]]
    #         for j in range(len(coordinates)):
    #             plt.scatter(coordinates[j, 0], coordinates[j, 1], s=circle_sizes[j], color=colors[j], alpha=0.5)
    #         plt.xlabel('X')
    #         plt.ylabel('Y')
    #         plt.title(f'Molecular Orbital {i+1}')
    #         plt.axis('equal')
    #         plt.grid(True)
    #         plt.show()
        
    def plot_energy_levels(self):
        x_values_benzene = [0, 0, 0.1, 0, 0.1, 0]
        y_values_benzene = [-12, -11, -11, -9, -9, -8]
        for i in range(len(x_values_benzene)):
            plt.hlines(y_values_benzene[i], xmin=x_values_benzene[i], xmax=x_values_benzene[i] + 0.08, color='blue')
        plt.margins(x=3)
        plt.ylabel('Energy (eV)')
        plt.title('Energy Levels of Benzene')
        plt.show()