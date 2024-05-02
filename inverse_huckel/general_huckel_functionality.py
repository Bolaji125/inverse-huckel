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
        #print (self.H)
        #print (np.linalg.eigh(self.H))
        return np.linalg.eigh(self.H)
    
    def plot_molecular_orbitals(self, coordinates,i, molecule_name = "Molecule"):
        energies, wavefunctions = self.solve_eigenvalue_problem()
        #print("energy of MO ", i+1, ":", energies[i])
        #print(energies)
        num_mos = len(wavefunctions[0]) #calculates the number of mos based on the length of the wavenfunctions array

        # Draw lines for pairs of atoms with short distances
        self.draw_short_distance_lines(coordinates) #calling the method below

        circle_sizes = np.abs(wavefunctions[:, i]) * 5000
        colors = ['green' if val >= 0 else 'red' for val in wavefunctions[:, i]]
        for j in range(len(coordinates)):
            plt.scatter(coordinates[j, 0], coordinates[j, 1], s=circle_sizes[j], color=colors[j], alpha=0.5)
        plt.xlabel('X coordinates/$\AA$')
        plt.ylabel('Y coordinates/$\AA$')
        plt.title( f'Molecular Orbital {i+1} of {molecule_name}')
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    def draw_short_distance_lines(self, coordinates):
        threshold_distance = 2  # rough distance between each coordinate pair
        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):
                distance = np.linalg.norm(coordinates[i] - coordinates[j])
                if distance <= threshold_distance:
                    plt.plot([coordinates[i, 0], coordinates[j, 0]], [coordinates[i, 1], coordinates[j, 1]], 'k-')

    def plot_energy_levels(self, energies, molecule_name="Molecule"):
        #print("Energies:", [f"{energy:.16f}" for energy in energies])  # Print the energies with full precision
        prev_energy = None #initialises previous enegy to none. it is used to keep track of the previous energy while iterating through the list of energies.
        x_offset = 0  # x offset is the starting point for drawing each horizontal line on the plot. setting it to 0 means that first energy level will be drawn from the leftmost side of the plot.
        tolerance = 1e-6  # Set a small tolerance value to determine if two energy levels are degenerate. if the absolute difference between two energy levels is less than this tolerance, they are considered degenerate
        for energy in energies: #starts a loop that iterates over each energy level in the list of energies
            if prev_energy is not None and abs(energy - prev_energy) < tolerance: #This condition checks if prev_energy is not None 
            #(i.e., if it's not the first energy level) and if the absolute difference between the current energy level (energy) and the previous energy level prev_energy is less than the tolerance value
                pass #if condition is met, the code does nothing. the current energy level is degeneartes with the previous energy level and the offset is not reset.
            else:
                # Otherwise, reset the x offset for non-degenerate energy levels
                x_offset = 0
            
            plt.hlines(energy, xmin=x_offset, xmax=0.1 + x_offset, color='blue')  # xmin and xmax determine the length of the horizontal line. 
            prev_energy = energy #this updates the prev_energy variable to store the current energy level for the next iteration of the loop
            x_offset += 0.3  # Increase the x offset for every plotted line
        
        plt.margins(x=3)
        plt.xlabel('Energy Levels') #is this label ok?
        plt.ylabel('Energy (eV)')
        plt.title(f'Energy Levels of {molecule_name}')
        plt.show()

  

