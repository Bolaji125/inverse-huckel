import numpy as np
import matplotlib.pyplot as plt

from general_huckel_functionality import MolecularSystem

# # Example usage
benzene_coordinates = np.array([
    [-4.461121, 1.187057, -0.028519],
    [-3.066650, 1.263428, -0.002700],
    [-2.303848, 0.094131, 0.041626],
    [-2.935547, -1.151550, 0.059845],
    [-4.330048, -1.227982, 0.034073],
    [-5.092743, -0.058655, -0.010193]
])

alpha = -10
beta = -1
cutoff_distance = 2.0
 #benzene
# molecular_system = MolecularSystem(benzene_coordinates, alpha, beta, cutoff_distance)
# molecular_system.plot_molecular_orbitals() #i didn't have to include this before i tried to change the code
# molecular_system.plot_energy_levels()

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
molecular_system.plot_molecular_orbitals(napthalene_coordinates)
#molecular_system.plot_energy_levels()


# molecular_system = MolecularSystem(benzene_coordinates, alpha, beta, cutoff_distance)
# molecular_system.plot_molecular_orbitals()

