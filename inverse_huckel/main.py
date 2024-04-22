#print("Hello World!")
#create a script that evaluates the huckel hamiltonian(solves it) 

import numpy as np

def huckel_hamiltonian(n, alpha, beta):
    """
    Constructs the Huckel Hamiltonian matrix for a cyclic pi system.

    Args:
    - n: Number of atoms in the cyclic pi system.
    - alpha: Alpha parameter (energy of each isolated atomic orbital).
    - beta: Beta parameter (resonance integral between adjacent atoms).

    Returns:
    - H: Huckel Hamiltonian matrix.
    """
    H = np.zeros((n, n))
    for i in range(n):
        H[i, i] = alpha
        H[(i+1) % n, i] = beta
        H[i, (i+1) % n] = beta
    return H

def solve_huckel_hamiltonian(n, alpha, beta):
    """
    Solves the Huckel Hamiltonian for a cyclic pi system.

    Args:
    - n: Number of atoms in the cyclic pi system.
    - alpha: Alpha parameter (energy of each isolated atomic orbital).
    - beta: Beta parameter (resonance integral between adjacent atoms).

    Returns:
    - energies: Eigenvalues of the Huckel Hamiltonian.
    - wavefunctions: Eigenvectors of the Huckel Hamiltonian.
    """
    H = huckel_hamiltonian(n, alpha, beta)
    energies, wavefunctions = np.linalg.eigh(H)
    return energies, wavefunctions

# Example usage:
n = 6  # Number of atoms
alpha = -1.0  # Energy of isolated atomic orbital
beta = -0.5  # Resonance integral between adjacent atoms

energies, wavefunctions = solve_huckel_hamiltonian(n, alpha, beta)

print("Eigenvalues:")
print(energies)
print("\nEigenvectors (each column represents a wavefunction):")
print(wavefunctions)

#workshop
#question 1
def allyl_radical_energy(alpha, beta):
    E1 = alpha + np.sqrt(2) * beta
    E2 = alpha
    E3 = alpha - np.sqrt(2) * beta
    return E1, E2, E3

def cyclopropenyl_radical_energy(alpha, beta):
    E1 = alpha + 2 * beta # can this also be called E1?
    E2 = alpha - beta
    E3 = alpha - beta
    return E1, E2, E3

# Constants for allyl radical
alpha_allyl = -1.0
beta_allyl = -0.5

# Constants for cyclopropenyl radical
alpha_cyclopropenyl = -1.0
beta_cyclopropenyl = -0.5

# Calculate molecular orbital energies for allyl radical
E_allyl = allyl_radical_energy(alpha_allyl, beta_allyl)
print("Molecular orbital energies for allyl radical:")
print("E_allyl1:", E_allyl[0])
print("E_allyl2:", E_allyl[1])
print("E_allyl3:", E_allyl[2])

# Calculate molecular orbital energies for cyclopropenyl radical
E_cyclopropenyl = cyclopropenyl_radical_energy(alpha_cyclopropenyl, beta_cyclopropenyl)
print("\nMolecular orbital energies for cyclopropenyl radical:")
print("E_cyclopropenyl1:", E_cyclopropenyl[0])
print("E_cyclopropenyl2:", E_cyclopropenyl[1])
print("E_cyclopropenyl3:", E_cyclopropenyl[2])

#question 4


