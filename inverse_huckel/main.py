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