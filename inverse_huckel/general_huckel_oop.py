import numpy as np
import matplotlib.pyplot as plt

from general_huckel_functionality import MolecularSystem

alpha = -10
beta = -1
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

molecular_system = MolecularSystem(benzene_coordinates, alpha, beta, cutoff_distance)
molecular_system.plot_molecular_orbitals(benzene_coordinates, 2, "Benzene")

energies_benzene = molecular_system.solve_eigenvalue_problem()[0]  # obtaining the energy, [0] extracts the first element of the tuple which corresponds to the eigenvalues
molecular_system.plot_energy_levels(energies_benzene, "Benzene")
print("benzene", energies_benzene)

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


molecular_system = MolecularSystem(napthalene_coordinates, alpha, beta, cutoff_distance)
molecular_system.plot_molecular_orbitals(napthalene_coordinates,5, "Napthalene")

energies_napthalene = molecular_system.solve_eigenvalue_problem()[0]
molecular_system.plot_energy_levels(energies_napthalene, "Napthalene")
print("napthalene", energies_napthalene)




