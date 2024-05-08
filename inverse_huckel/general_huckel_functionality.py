import numpy as np
import matplotlib.pyplot as plt
import os
import torch

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
    
    def solve_eigenvalue_problem_pytorch(self):
        H = self.construct_hamiltonian()
        H = torch.from_numpy(H)
        # Perform eigenvalue decomposition using PyTorch
        eigenvalues_complex, eigenvectors = torch.linalg.eig(H)
        #eigenvalues_complex = torch.linalg.eigvals(H)
                
        # Extract the real part of eigenvalues
        #eigenvalues_real = eigenvalues_complex.numpy()
        eigenvalues_real = eigenvalues_complex.real

        return eigenvalues_real, eigenvectors
    
    def plot_molecular_orbitals(self, coordinates,i, molecule_name = "Molecule"):
        energies, wavefunctions = self.solve_eigenvalue_problem()
        #print("energy of MO ", i+1, ":", energies[i])
        #print(energies)
        num_mos = len(wavefunctions[0]) #calculates the number of mos based on the length of the wavenfunctions array
        fig = plt.figure()
        # Draw lines for pairs of atoms with short distances
        self.draw_short_distance_lines(coordinates) #calling the method below

        circle_sizes = np.abs(wavefunctions[:, i]) * 5000
        colors = ['green' if val >= 0 else 'red' for val in wavefunctions[:, i]]
        for j in range(len(coordinates)):
            plt.scatter(coordinates[j, 0], coordinates[j, 1], s=circle_sizes[j], color=colors[j], alpha=0.5)
        plt.xlabel('X / $\AA$')
        plt.ylabel('Y / $\AA$')
        plt.title( f'Molecular Orbital {i+1} of {molecule_name}')
        plt.axis('equal')
        plt.grid(True)
        file_name = f'Molecular Orbital {i+1} of {molecule_name}.pdf'
        file_path = os.path.join("C:\\Users\\ogunn\\Documents\\GitHub\\inverse-huckel.git\\inverse_huckel", file_name)
        fig.savefig(file_path)
        plt.show()

    def draw_short_distance_lines(self, coordinates):
        threshold_distance = 2  # rough distance between each coordinate pair
        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):
                distance = np.linalg.norm(coordinates[i] - coordinates[j])
                if distance <= threshold_distance:
                    plt.plot([coordinates[i, 0], coordinates[j, 0]], [coordinates[i, 1], coordinates[j, 1]], 'k-')

    def plot_molecular_orbitals_pytorch(self, coordinates, i, molecule_name="Molecule"):
        energies, wavefunctions = self.solve_eigenvalue_problem_pytorch()

        fig = plt.figure()
        # Draw lines for pairs of atoms with short distances
        self.draw_short_distance_lines(coordinates)

        circle_sizes = torch.abs(wavefunctions[:, i]) * 5000  # Use absolute value of the corresponding wavefunctions
        # Extract the real part of wavefunctions
        real_wavefunctions = wavefunctions[:, i].real
        # Use real part for color instead of direct comparison
        colors = ['green' if val >= 0 else 'red' for val in real_wavefunctions]
        for j in range(len(coordinates)):
            plt.scatter(coordinates[j, 0], coordinates[j, 1], s=circle_sizes[j], color=colors[j], alpha=0.5)

        plt.xlabel('X / $\AA$')
        plt.ylabel('Y / $\AA$')
        plt.title(f'Molecular Orbital {i+1} of {molecule_name}')
        plt.axis('equal')
        plt.grid(True)
        file_name = f'Molecular Orbital {i+1} of {molecule_name}_pytorch.pdf'
        file_path = os.path.join("C:\\Users\\ogunn\\Documents\\GitHub\\inverse-huckel.git\\inverse_huckel", file_name)
        plt.show()


    def plot_energy_levels(self, energies, molecule_name="Molecule"): #isn't reproducable for pytorch because enrgies aren't ordered
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
        plt.xlabel('Energy Levels') 
        plt.ylabel('Energy (eV)')
        plt.title(f'Energy Levels of {molecule_name}')

         # Define the file path and name based on the molecule name
        file_name = f"{molecule_name.replace(' ', '_')}_energy_levels.pdf"
        file_path = os.path.join("C:\\Users\\ogunn\\Documents\\GitHub\\inverse-huckel.git\\inverse_huckel", file_name)
        
        # Save the plot
        plt.savefig(file_path)

        plt.show()


    def plot_energy_levels_pytorch(self, energies, molecule_name="Molecule"): #works but benezene energy level ver small.doesn't work for napthalene
        eigenvalues, _ = energies  # Unpack the tuple, _ is a paceholder to ignore the second element of the tuple as it is not needed
        #eigenvalues = torch.tensor(eigenvalues)  # Convert eigenvalues to a PyTorch tensor
        eigenvalues = eigenvalues.clone().detach()
        print("eigenvalues", eigenvalues)

        prev_energy = None #keeps track of previous energy level
        x_offset = 0   #determines the starting position of each energy level line on the plot
        #tolerance = 1e-4 #tolerance determines if two energy levels are considered degenerate

        for energy in eigenvalues.numpy():  # Convert eigenvalues to a NumPy array to iterate over the energy levels
            # if prev_energy is not None and torch.abs(torch.tensor(energy) - torch.tensor(prev_energy)) < tolerance:
            #     pass 
            # else:
            #     x_offset = 0

            plt.hlines(energy, xmin=x_offset, xmax=0.3 + x_offset, color='blue')  
            prev_energy = energy 
            x_offset += 0.5  #line determines energy spacing between consecutive energy levels

        plt.margins(x=3)
        plt.xlabel('Energy Levels') 
        plt.ylabel('Energy (eV)')
        plt.title(f'Energy Levels of {molecule_name}')
        plt.show()

    # def plot_energy_levels_pytorch(self, energies, molecule_name="Molecule"): #doesn't work correctly as 5 levels instead of 6.
    #     eigenvalues, _ = energies  # Unpack the tuple
    #     eigenvalues = torch.tensor(eigenvalues).clone().detach()

    #     prev_energy = None 
    #     x_offset = 0   # Determine the starting position of each energy level line on the plot
    #     spacing = 0.3  # Set the spacing between consecutive energy levels
    #     degenerate_spacing = 0.1  # Set the additional spacing for degenerate energy levels

    #     for energy in eigenvalues.numpy():  # Convert eigenvalues to a NumPy array to iterate over the energy levels
    #         if prev_energy is not None and energy == prev_energy:
    #             x_offset += degenerate_spacing
    #         else:
    #             x_offset += spacing

    #         plt.hlines(energy, xmin=x_offset, xmax=spacing + x_offset, color='blue')  
    #         prev_energy = energy 

    #     plt.margins(x=3)
    #     plt.xlabel('Energy Levels') 
    #     plt.ylabel('Energy (eV)')
    #     plt.title(f'Energy Levels of {molecule_name}')
    #     plt.show()

    # def plot_energy_levels_pytorch(self, energies, molecule_name="Molecule"): #works but doesn't look right. see week 3 notes for logic.
    #     eigenvalues, _ = energies  # Unpack the tuple
    #     eigenvalues = torch.tensor(eigenvalues).clone().detach()

    #     prev_energy = None 
    #     x_offset = 0   # Determine the starting position of each energy level line on the plot
    #     line_length = 0.2  # Set the length of each energy level line

    #     for energy in eigenvalues.numpy():  # Convert eigenvalues to a NumPy array to iterate over the energy levels
    #         plt.hlines(energy, xmin=x_offset, xmax=x_offset + line_length, color='blue')  
    #         prev_energy = energy 

    #     plt.xlabel('Energy Levels') 
    #     plt.ylabel('Energy (eV)')
    #     plt.title(f'Energy Levels of {molecule_name}')
    #     plt.show()

    

    
# Define the directory where you want to save the file
# directory = 'Documents/GitHub/inverse-huckel.git/inverse_huckel'

# # Construct the full file path
# file_path = os.path.join(os.path.expanduser('~'), directory)

# # Print the file path
# print("File path:", file_path)
