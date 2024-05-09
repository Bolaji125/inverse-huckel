import numpy as np
import matplotlib.pyplot as plt

from general_huckel_functionality import MolecularSystem
import torch

alpha = -10.0
beta = -1.0
cutoff_distance = 2.0

benzene_coordinates = np.array([
    [-4.461121, 1.187057, -0.028519],
    [-3.066650, 1.263428, -0.002700],
    [-2.303848, 0.094131, 0.041626],
    [-2.935547, -1.151550, 0.059845],
    [-4.330048, -1.227982, 0.034073],
    [-5.092743, -0.058655, -0.010193]
])


 #benzene
 # create molecular system instance
molecular_system = MolecularSystem(benzene_coordinates, alpha, beta, cutoff_distance)

#plot molecular orbitals
# molecular_system.plot_molecular_orbitals(benzene_coordinates, 5, "Benzene")

# # plot energy levels
# energies_benzene = molecular_system.solve_eigenvalue_problem()[0]  # obtaining the energy, [0] is used to access the first element of the tuple which corresponds to the eigenvalues
# molecular_system.plot_energy_levels(energies_benzene, "Benzene")
# print("benzene", energies_benzene)

# #obtain energy
# energies_pytorch = molecular_system.solve_eigenvalue_problem_pytorch()

# # plot molecular orbitals - pytorch
# molecular_system.plot_molecular_orbitals_pytorch(benzene_coordinates, 0, "Benzene")

# #plot energy levels - pytorch
# molecular_system.plot_energy_levels_pytorch(energies_pytorch, "Benzene")
# print("energies", energies_pytorch)

# target eigenvalues- example
target_eigenvalues_benzene = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

gradient_benzene = molecular_system.compute_gradient_with_respect_to_eigenvalues(target_eigenvalues_benzene)

# Compute and visualise gradient
molecular_system.visualise_gradient(target_eigenvalues_benzene, "Benzene")



#napthalene

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

# create molecular system instance
molecular_system = MolecularSystem(napthalene_coordinates, alpha, beta, cutoff_distance)

# plot molecular orbitals
# molecular_system.plot_molecular_orbitals(napthalene_coordinates,9, "Napthalene")

# obtain energy
#energies_napthalene = molecular_system.solve_eigenvalue_problem()[0]

# plot energy levels
#molecular_system.plot_energy_levels(energies_napthalene, "Napthalene")

#obtain energy - pytorch
#energies_pytorch = molecular_system.solve_eigenvalue_problem_pytorch()

#plot molecular orbitals- pytorch
# #molecular_system.plot_molecular_orbitals_pytorch(napthalene_coordinates, 0, "Napthalene")

#plot energy levels-pytorch
#molecular_system.plot_energy_levels_pytorch(energies_pytorch, "Napthalene")

# target eigenvalues- example
target_eigenvalues_napthalene = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

gradient_napthalene = molecular_system.compute_gradient_with_respect_to_eigenvalues(target_eigenvalues_napthalene)

# Compute and visualise gradient
molecular_system.visualise_gradient(target_eigenvalues_napthalene, "Napthalene")



# coordinates = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
# system = MolecularSystem(coordinates, alpha, beta, cutoff_distance)
# system.visualise_gradient()




