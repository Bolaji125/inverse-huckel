import numpy as np

class HuckelHamiltonian:
    def __init__(self, n, alpha, beta):
        self.H = self.setup_hamiltonian(n, alpha, beta)
#make the hamiltonian more general for all molecules
    def setup_hamiltonian(self, n, alpha, beta):
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

    def solve_huckel_hamiltonian(self):
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
        #H = huckel_hamiltonian(n, alpha, beta)
        energies, wavefunctions = np.linalg.eigh(self.H)
        return energies, wavefunctions