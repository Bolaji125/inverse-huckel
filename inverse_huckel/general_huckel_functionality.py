import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from matplotlib import cm

class MolecularSystem:

    def __init__(self, coordinates, alpha, beta, cutoff_distance):
        self.coordinates = coordinates
        self.alpha = torch.tensor(alpha)
        self.beta = beta
        self.cutoff_distance = cutoff_distance
        self.H = self.construct_hamiltonian()
    # def __init__(self, coordinates, alpha, beta, cutoff_distance):
    #     self.coordinates = coordinates
    #     self.alpha = torch.tensor(alpha, requires_grad=True)  # Set requires_grad to True
    #     self.beta = torch.tensor(beta, requires_grad=True)  # Set requires_grad to True
    #     self.cutoff_distance = cutoff_distance
    #     self.H = self.construct_hamiltonian()

    def calculate_distance(self, atom_i, atom_j):
        atom_i_tensor = torch.tensor(atom_i)
        atom_j_tensor = torch.tensor(atom_j)
        return torch.linalg.norm(atom_i_tensor - atom_j_tensor)

    def construct_hamiltonian(self):
        n = len(self.coordinates)
        H = torch.zeros((n, n))  # create a new tensor for H
        for i in range(n):
            for j in range(i + 1, n):
                distance_ij = self.calculate_distance(self.coordinates[i], self.coordinates[j])
                if distance_ij < self.cutoff_distance:
                    H[i, j] = self.beta
                    H[j, i] = self.beta
        H.fill_diagonal_(self.alpha.item())  # Fill diagonal with scalar value of alpha
        return H

    # def update_hamiltonian(self):
    #     self.H = self.construct_hamiltonian()

    def solve_eigenvalue_problem(self):
        #print (self.H)
        #print (np.linalg.eigh(self.H))
        # H_numpy = self.H.detach().numpy()  # Detach the tensor before converting to NumPy array
        # return np.linalg.eigh(H_numpy)

        return np.linalg.eigh(self.H)
    
    
    def solve_eigenvalue_problem_pytorch(self):
        eigenvalues, eigenvectors = torch.linalg.eigh(self.H)
        return eigenvalues, eigenvectors
    
    def plot_molecular_orbitals(self, coordinates, molecule_name="Molecule"):
        energies, wavefunctions = self.solve_eigenvalue_problem()
        num_mos = len(wavefunctions[0])  # Calculate the number of molecular orbitals

        # Calculate the number of rows and columns for the subplot grid
        num_cols = min(num_mos, 3)  # Maximum of 3 columns
        num_rows = (num_mos + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

        max_circle_size = 2500 * np.max(np.abs(wavefunctions))
        margin = max_circle_size ** 0.5 / 2  # Calculate the margin to fit the largest circle

        min_x, max_x = np.min(coordinates[:, 0]) - margin, np.max(coordinates[:, 0]) + margin
        min_y, max_y = np.min(coordinates[:, 1]) - margin, np.max(coordinates[:, 1]) + margin

        for i in range(num_mos):
            row = i // num_cols
            col = i % num_cols

            circle_sizes = np.abs(wavefunctions[:, i]) * 5000
            colors = ['green' if val >= 0 else 'red' for val in wavefunctions[:, i]]

            for j in range(len(coordinates)):
                axes[row, col].scatter(coordinates[j, 0], coordinates[j, 1], s=circle_sizes[j], color=colors[j], alpha=0.5)

            # Draw lines for pairs of atoms with short distances
            self.draw_short_distance_lines(coordinates, axes[row, col])

            axes[row, col].set_xlim(min_x, max_x)
            axes[row, col].set_ylim(min_y, max_y)
            axes[row, col].set_xlabel('X / $\AA$', fontsize=7)
            axes[row, col].set_ylabel('Y / $\AA$',fontsize=7)
            axes[row, col].set_title(f'Molecular Orbital {i+1} of {molecule_name} using NumPy', fontsize=10)
            axes[row, col].axis('equal')
            axes[row, col].grid(True)

        # Remove any unused subplots
        for ax in axes.flatten()[num_mos:]:
            fig.delaxes(ax)

    
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to prevent overlap
        file_name = f'{molecule_name}_molecular_orbitals.png'
        file_path = os.path.join("C:\\Users\\ogunn\\Documents\\GitHub\\inverse-huckel.git\\inverse_huckel", file_name)
        plt.savefig(file_path)
        plt.show()
        plt.close(fig)  # Close the figure to free up memory

    
    def draw_short_distance_lines(self, coordinates, ax):
        threshold_distance = 1.5  # rough distance between each coordinate pair
        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):
                distance = np.linalg.norm(coordinates[i] - coordinates[j])
                if distance <= threshold_distance:
                    ax.plot([coordinates[i, 0], coordinates[j, 0]], [coordinates[i, 1], coordinates[j, 1]], 'k-')


    # def plot_molecular_orbitals(self, coordinates,i, molecule_name = "Molecule"): # old code
    #     energies, wavefunctions = self.solve_eigenvalue_problem()
    #     #print("energy of MO ", i+1, ":", energies[i])
    #     #print(energies)
    #     num_mos = len(wavefunctions[0]) #calculates the number of mos based on the length of the wavenfunctions array
    #     fig = plt.figure()
    #     # Draw lines for pairs of atoms with short distances
    #     self.draw_short_distance_lines(coordinates) #calling the method below

    #     circle_sizes = np.abs(wavefunctions[:, i]) * 5000
    #     colors = ['green' if val >= 0 else 'red' for val in wavefunctions[:, i]]
    #     for j in range(len(coordinates)):
    #         plt.scatter(coordinates[j, 0], coordinates[j, 1], s=circle_sizes[j], color=colors[j], alpha=0.5)
    #     plt.xlabel('X / $\AA$')
    #     plt.ylabel('Y / $\AA$')
    #     plt.title( f'Molecular Orbital {i+1} of {molecule_name} using NumPy')
    #     plt.axis('equal')
    #     plt.grid(True)
    #     file_name = f'Molecular Orbital {i+1} of {molecule_name}.png'
    #     file_path = os.path.join("C:\\Users\\ogunn\\Documents\\GitHub\\inverse-huckel.git\\inverse_huckel", file_name)
    #     #fig.savefig(file_path)
    #     plt.savefig(file_path)
    #     plt.show()
    #     plt.close(fig)  # Close the figure to free up memory


    # def draw_short_distance_lines(self, coordinates): # old code
    #     threshold_distance = 2  # rough distance between each coordinate pair
    #     for i in range(len(coordinates)):
    #         for j in range(i + 1, len(coordinates)):
    #             distance = np.linalg.norm(coordinates[i] - coordinates[j])
    #             if distance <= threshold_distance:
    #                 plt.plot([coordinates[i, 0], coordinates[j, 0]], [coordinates[i, 1], coordinates[j, 1]], 'k-')

    def plot_molecular_orbitals_pytorch(self, coordinates, molecule_name="Molecule"):
        energies, wavefunctions = self.solve_eigenvalue_problem_pytorch()
        num_mos = wavefunctions.shape[1]  # Calculate the number of molecular orbitals

        # Calculate the number of rows and columns for the subplot grid
        num_cols = min(num_mos, 3)  # Maximum of 3 columns
        num_rows = (num_mos + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(7 * num_cols, 7 * num_rows))

        max_circle_size = 5000 * torch.max(torch.abs(wavefunctions)).item()
        margin = max_circle_size ** 0.5 / 2  # Calculate the margin to fit the largest circle

        min_x, max_x = np.min(coordinates[:, 0]) - margin, np.max(coordinates[:, 0]) + margin
        min_y, max_y = np.min(coordinates[:, 1]) - margin, np.max(coordinates[:, 1]) + margin

        for i in range(num_mos):
            row = i // num_cols
            col = i % num_cols

            circle_sizes = torch.abs(wavefunctions[:, i]) * 5000
            real_wavefunctions = wavefunctions[:, i].real
            colors = ['green' if val >= 0 else 'red' for val in real_wavefunctions]

            for j in range(len(coordinates)):
                axes[row, col].scatter(coordinates[j, 0], coordinates[j, 1], s=circle_sizes[j].item(), color=colors[j], alpha=0.5)

            # Draw lines for pairs of atoms with short distances
            self.draw_short_distance_lines(coordinates, axes[row, col])

            axes[row, col].set_xlim(min_x, max_x)
            axes[row, col].set_ylim(min_y, max_y)
            axes[row, col].set_xlabel('X / $\AA$', fontsize=10)
            axes[row, col].set_ylabel('Y / $\AA$', fontsize=10)
            axes[row, col].set_title(f'Molecular Orbital {i+1} of {molecule_name} using PyTorch', fontsize=12)
            axes[row, col].axis('equal')
            axes[row, col].grid(True)

        # Remove any unused subplots
        for ax in axes.flatten()[num_mos:]:
            fig.delaxes(ax)

        plt.subplots_adjust(hspace=0.5, wspace=0.4, bottom=0.1, top=0.9)  # Adjust the spacing
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to prevent overlap

        file_name = f'{molecule_name}_molecular_orbitals_pytorch.png'
        file_path = os.path.join("C:\\Users\\ogunn\\Documents\\GitHub\\inverse-huckel.git\\inverse_huckel", file_name)
        plt.savefig(file_path)
        plt.show()
   
    # def plot_molecular_orbitals_pytorch(self, coordinates, i, molecule_name="Molecule"): # old code
    #     energies, wavefunctions = self.solve_eigenvalue_problem_pytorch()

    #     fig = plt.figure()
    #     # Draw lines for pairs of atoms with short distances
    #     self.draw_short_distance_lines(coordinates)

    #     circle_sizes = torch.abs(wavefunctions[:, i]) * 5000  # Use absolute value of the corresponding wavefunctions
    #     # Extract the real part of wavefunctions
    #     real_wavefunctions = wavefunctions[:, i].real
    #     # Use real part for color instead of direct comparison
    #     colors = ['green' if val >= 0 else 'red' for val in real_wavefunctions]
    #     for j in range(len(coordinates)):
    #         plt.scatter(coordinates[j, 0], coordinates[j, 1], s=circle_sizes[j], color=colors[j], alpha=0.5)

    #     plt.xlabel('X / $\AA$')
    #     plt.ylabel('Y / $\AA$')
    #     plt.title(f'Molecular Orbital {i+1} of {molecule_name} using PyTorch')
    #     plt.axis('equal')
    #     plt.grid(True)
        
    #     # save plot as PNG file
    #     file_name = f'Molecular Orbital {i+1} of {molecule_name}_pytorch.png'
    #     file_path = os.path.join("C:\\Users\\ogunn\\Documents\\GitHub\\inverse-huckel.git\\inverse_huckel", file_name)
    #     plt.show()
    #     plt.savefig(file_path)
    #     plt.close(fig)  # Close the figure to free up memory


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
        plt.title(f'Energy Levels of {molecule_name} using NumPy')

        # Define the file path and name based on the molecule name
        file_name = f"{molecule_name.replace(' ', '_')}_energy_levels.png"
        file_path = os.path.join("C:\\Users\\ogunn\\Documents\\GitHub\\inverse-huckel.git\\inverse_huckel", file_name)
        
        # Save the plot
        plt.savefig(file_path)

        plt.show()

        plt.close()  # Close the figure to free up memory


    def plot_energy_levels_pytorch(self, eigenvalues, molecule_name="Molecule"): # png save code that doesn't pick up all degenerate energy levels
        eigenvalues = eigenvalues.clone().detach()
        print("original eigenvalues", eigenvalues)

        prev_energy = None  # keeps track of the previous energy level
        x_offset = 0  # determines the starting position of each energy level line on the plot
        tolerance = 1e-6  # tolerance to determine if two energy levels are degenerate

        for energy in eigenvalues.numpy():  # Convert eigenvalues to a NumPy array to iterate over the energy levels
            print("current energy", energy)
            if prev_energy is not None and abs(energy - prev_energy) < tolerance:
                x_offset += 0.3  # Increase the x offset to add whitespace for degenerate levels
            else:
                x_offset = 0  # Reset x offset for non-degenerate energy levels

            plt.hlines(energy, xmin=x_offset, xmax=0.3 + x_offset, color='blue')  
            prev_energy = energy  # Update the previous energy level
            x_offset += 0.5  # Add spacing between consecutive energy levels
        print("final eigenvalues", eigenvalues)

        plt.margins(x=3)
        plt.xlabel('Energy Levels') 
        plt.ylabel('Energy (eV)')
        plt.title(f'Energy Levels of {molecule_name} using PyTorch')

        # Define the file path and name based on the molecule name
        file_name = f"{molecule_name.replace(' ', '_')}_energy_levels_pytorch.png"
        file_path = os.path.join("C:\\Users\\ogunn\\Documents\\GitHub\\inverse-huckel.git\\inverse_huckel", file_name)
        
        # Save the plot
        plt.savefig(file_path)
        plt.show()
        plt.close()  # Close the figure to free up memory

       
        
    # def plot_energy_levels_pytorch(self, energies, molecule_name="Molecule"): # previous code, 
    #     #eigenvalues, _ = energies  # Unpack the tuple, _ is a placeholder to ignore the second element of the tuple as it is not needed
    #     eigenvalues = eigenvalues.clone().detach()
    #     print("eigenvalues", eigenvalues)

    #     prev_energy = None  # keeps track of the previous energy level
    #     x_offset = 0  # determines the starting position of each energy level line on the plot
    #     tolerance = 1e-6  # tolerance to determine if two energy levels are degenerate

    #     for energy in eigenvalues.numpy():  # Convert eigenvalues to a NumPy array to iterate over the energy levels
    #         if prev_energy is not None and abs(energy - prev_energy) < tolerance:
    #             x_offset += 0.3  # Increase the x offset to add whitespace for degenerate levels
    #         else:
    #             x_offset = 0  # Reset x offset for non-degenerate energy levels

    #         plt.hlines(energy, xmin=x_offset, xmax=0.3 + x_offset, color='blue')  
    #         prev_energy = energy  # Update the previous energy level
    #         x_offset += 0.5  # Add spacing between consecutive energy levels

    #     plt.margins(x=3)
    #     plt.xlabel('Energy Levels') 
    #     plt.ylabel('Energy (eV)')
    #     plt.title(f'Energy Levels of {molecule_name} using PyTorch')
        
    #     # Define the file path and name based on the molecule name
    #     # file_name = f"{molecule_name.replace(' ', '_')}_energy_levels_PyTorch.pdf"
    #     # file_path = os.path.join("C:\\Users\\ogunn\\Documents\\GitHub\\inverse-huckel.git\\inverse_huckel", file_name)
        
    #     # # Save the plot
    #     # plt.savefig(file_path)

    #     plt.show()


    
    
       

    

