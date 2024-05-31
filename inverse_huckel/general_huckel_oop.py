import numpy as np
import matplotlib.pyplot as plt

from general_huckel_functionality import MolecularSystem
import torch
import torch.linalg as linalg

alpha = -10
beta = -1.0
cutoff_distance = 2.0 

benzene_coordinates = np.array([ # scaled benzene coordinates
    [ 1.3965    ,  0.        ,  0.        ],
    [ 0.69825   ,  1.2095694 ,  0.        ],
    [-0.69825   ,  1.2095694 ,  0.        ],
    [-1.3965    ,  0.        ,  0.        ],
    [-0.69825   , -1.2095694 ,  0.        ],
    [ 0.69825   , -1.2095694 ,  0.        ]
])
 #benzene
 # create molecular system instance
molecular_system_benzene = MolecularSystem(benzene_coordinates, alpha, beta, cutoff_distance)

#plot molecular orbitals
#molecular_system_benzene.plot_molecular_orbitals(benzene_coordinates,"Benzene")

# plot energy levels
#energies_benzene = molecular_system_benzene.solve_eigenvalue_problem()[0]  # obtaining the energy, [0] is used to access the first element of the tuple which corresponds to the eigenvalues
#molecular_system_benzene.plot_energy_levels(energies_benzene, "Benzene")

#obtain energy- pytorch
energies_pytorch, eigenvectors_pytorch = molecular_system_benzene.solve_eigenvalue_problem_pytorch()

## plot molecular orbitals - pytorch
molecular_system_benzene.plot_molecular_orbitals_pytorch(benzene_coordinates,"Benzene")

# #plot energy levels - pytorch
molecular_system_benzene.plot_energy_levels_pytorch(energies_pytorch, "Benzene")


#naphthalene

naphthalene_coordinates = np.array([
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
molecular_system_naphthalene = MolecularSystem(naphthalene_coordinates, alpha, beta, cutoff_distance)

# plot molecular orbitals
#molecular_system_naphthalene.plot_molecular_orbitals(naphthalene_coordinates,"Naphthalene")

# obtain energy
#energies_naphthalene = molecular_system_naphthalene.solve_eigenvalue_problem()[0]
#print("energies naphthalene:", energies_naphthalene)

# plot energy levels
#molecular_system_naphthalene.plot_energy_levels(energies_naphthalene, "Naphthalene")

#obtain energy - pytorch
energies_pytorch = molecular_system_naphthalene.solve_eigenvalue_problem_pytorch()
energies_pytorch, eigenvectors_pytorch = molecular_system_naphthalene.solve_eigenvalue_problem_pytorch()

#plot molecular orbitals- pytorch
molecular_system_naphthalene.plot_molecular_orbitals_pytorch(naphthalene_coordinates,"Naphthalene")

#plot energy levels-pytorch
molecular_system_naphthalene.plot_energy_levels_pytorch(energies_pytorch, "Naphthalene")

