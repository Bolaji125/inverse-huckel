import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

class MolecularSystem:
    def __init__(self, coordinates, alpha, beta, cutoff_distance):
        self.coordinates = torch.tensor(coordinates, dtype=torch.float32)
        self.alpha = torch.tensor([alpha] * len(coordinates), requires_grad=True)
        self.cutoff_distance = cutoff_distance

        # Calculate the number of valid beta values needed
        self.beta_indices = []
        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):
                distance = np.linalg.norm(coordinates[i] - coordinates[j])
                if distance <= cutoff_distance:
                    self.beta_indices.append((i, j))
        
        self.beta = torch.tensor([beta] * len(self.beta_indices), requires_grad=True)
        self.H = torch.zeros((len(coordinates), len(coordinates)), dtype=torch.float32)
        self.update_hamiltonian()  # calls update hamiltonian to initialize the matrix


    def calculate_distance(self, atom_i, atom_j): 
        #Calculate the Euclidean distance between two atoms
        atom_i_tensor = torch.tensor(atom_i)
        atom_j_tensor = torch.tensor(atom_j)
        return torch.linalg.norm(atom_i_tensor - atom_j_tensor)
    
    def update_hamiltonian(self):
        num_atoms = self.coordinates.shape[0]  # determines the number of atoms
        self.H = torch.zeros((num_atoms, num_atoms), dtype=torch.float32)  # initializes the Hamiltonian matrix H with zeros.
        
        # Set diagonal elements to alpha values
        for i in range(num_atoms):
            self.H[i, i] = self.alpha[i]
        
        # Set off-diagonal elements based on cutoff distance and beta
        for pair_idx, (i, j) in enumerate(self.beta_indices):
            self.H[i, j] = self.beta[pair_idx]
            self.H[j, i] = self.beta[pair_idx]

    def solve_eigenvalue_problem_pytorch(self):
        # Solve the eigenvalue problem for the Hamiltonian using PyTorch.
        eigenvalues, eigenvectors = torch.linalg.eigh(self.H)
        return eigenvalues, eigenvectors

    
def optimise_molecular_system(molecular_system, target_eigenvalues, num_iterations=1000, learning_rate=0.01):
    # optimise the molecular system's parameters to match target eigenvalues using gradient descent
    alpha_history, beta_history, loss_history = [], [], [] #initialises an empty list to store the history of alpha, beta and loss values at each iteration
    eigenvalues, _ = molecular_system.solve_eigenvalue_problem_pytorch()
    print("eigenvalues before optimisation:", eigenvalues)
    print("hamiltonian:", molecular_system.H)

    for i in range(num_iterations + 1):
        molecular_system.update_hamiltonian() #updates hamiltonian matrix based on the current alpha and beta values
        eigenvalues, _ = molecular_system.solve_eigenvalue_problem_pytorch()
        loss = F.mse_loss(eigenvalues, target_eigenvalues) # Computes the mean squared error (MSE) loss between the calculated eigenvalues and the target eigenvalues.
        
        if i % 500 == 0 or i == num_iterations:
            print(f"Iteration {i}, Loss: {loss.item()}")
          

        if i < num_iterations:# Perform backpropagation to compute gradients if not the last iteration
            loss.backward() # performs backpropagation, which calculates the gradients of the loss function with respect to the model parameters (alpha and beta in this case).
        
        # gradient descent
        with torch.no_grad(): # Temporarily disables gradient tracking (since we are manually updating parameters).
            molecular_system.alpha -= learning_rate * molecular_system.alpha.grad # Updates the alpha parameters by subtracting the product of the learning rate and the gradients of alpha.
            molecular_system.beta -= learning_rate * molecular_system.beta.grad
        
        molecular_system.alpha.grad.zero_() # Resets the gradients of alpha to zero for the next iteration
        molecular_system.beta.grad.zero_()
        
        alpha_history.append(molecular_system.alpha.clone().detach().numpy()) # Records the current alpha values by appending a copy of the alpha tensor (converted to a NumPy array) to the alpha_history list.
        beta_history.append(molecular_system.beta.clone().detach().numpy()) 
        loss_history.append(loss.item())
        
    # Print the optimised alpha and beta values
    print("Optimised Alpha Values:", molecular_system.alpha)
    
    print("Optimised Beta Values:", molecular_system.beta)
   
    print("Optimised Hamiltonian:", molecular_system.H)
    eigenvalues_after, _ = molecular_system.solve_eigenvalue_problem_pytorch()
    
    print("Eigenvalues after optimisation:", eigenvalues_after.real)
    return np.array(alpha_history), np.array(beta_history), np.array(loss_history)

def generate_atom_pairs(coordinates, cutoff_distance): 
    # function to generate atom pairs
    pairs = [] # Initialises an empty list called pairs that will store the pairs of atom indices that are within the cutoff distance.
    num_atoms = len(coordinates)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = np.linalg.norm(coordinates[i] - coordinates[j])
            if distance <= cutoff_distance:
                pairs.append((i, j))
    return pairs

def plot_parameter_changes(alpha_history, beta_history, loss_history, molecule_name, beta_indices):
    epochs = np.arange(alpha_history.shape[0])

    fig, axs = plt.subplots(3, 1, figsize=(14, 20))

    # Plot alpha values
    for i in range(alpha_history.shape[1]):
        axs[0].plot(epochs, alpha_history[:, i], label=f'Alpha Atom {i+1}')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Alpha Value')
    axs[0].set_title(f'Alpha Parameter Changes for {molecule_name}')
    axs[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1.15), ncol=1)
    axs[0].grid(True)

    # Plot beta values for all pairs
    for i in range(len(beta_history[0])):
        axs[1].plot(epochs, beta_history[:, i], label=f'Beta Atom {beta_indices[i][0]+1}-{beta_indices[i][1]+1}')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Beta Value')
    axs[1].set_title(f'Beta Parameter Changes for {molecule_name}')
    axs[1].legend(loc='upper left', bbox_to_anchor=(1.02, 1), ncol=1)
    axs[1].grid(True)

    # Plot loss values
    axs[2].plot(epochs, loss_history, label='Loss Value', color='red')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Loss')
    axs[2].set_title(f'Loss Changes for {molecule_name}')
    axs[2].legend(loc='upper left', bbox_to_anchor=(1.02, 0.70))
    axs[2].grid(True)

    plt.subplots_adjust(right=0.75, hspace=0.6)
    plt.show()

def plot_molecule(coordinates, alpha_values, beta_values, molecule_name, cutoff_distance, ax=None): # compare to other plotting function
    if ax is None: # what is ax?
        fig, ax = plt.subplots(figsize=(8, 8))

    for i in range(len(coordinates)):
        alpha = alpha_values[i].item() if isinstance(alpha_values[i], torch.Tensor) else alpha_values[i]
        ax.scatter(coordinates[i, 0], coordinates[i, 1], s=1000 * abs(alpha), c='blue', alpha=0.5)
        ax.annotate(f'α{i+1}={alpha:.2f}', (coordinates[i, 0], coordinates[i, 1]))

    pair_idx = 0
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            distance = np.linalg.norm(coordinates[i] - coordinates[j])
            if distance <= cutoff_distance:
                beta = beta_values[pair_idx].item() if isinstance(beta_values[pair_idx], torch.Tensor) else beta_values[pair_idx]
                ax.plot([coordinates[i, 0], coordinates[j, 0]], [coordinates[i, 1], coordinates[j, 1]],
                        linewidth=abs(beta), color='black', alpha=0.5)
                midpoint = (coordinates[i] + coordinates[j]) / 2
                ax.annotate(f'β={beta:.2f}', midpoint[:2])  # Extract only x and y coordinates for annotation
                pair_idx += 1

    ax.set_aspect('equal')
    ax.set_title(f'Molecule Visualisation - {molecule_name}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    if ax is None:
        plt.show()


def optimise_and_plot(molecular_system, target_eigenvalues, molecule_name, cutoff_distance):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    alpha_history, beta_history, loss_history = optimise_molecular_system(molecular_system, target_eigenvalues)
    
    plot_molecule(molecular_system.coordinates, alpha_history[0], beta_history[0],
                  f"{molecule_name} - Before Optimisation", cutoff_distance, ax=axs[0])
    plot_molecule(molecular_system.coordinates, alpha_history[-1], beta_history[-1],
                  f"{molecule_name} - After Optimisation", cutoff_distance, ax=axs[1])
    
    plot_parameter_changes(alpha_history, beta_history, loss_history, molecule_name, molecular_system.beta_indices)
    
    plt.show()



alpha_initial = -10.0
beta_initial = -1.0
cutoff_distance = 2.0

target_eigenvalues_benzene = torch.tensor([-13.0, -12.0, -12.0, -9.0, -10.0, -9.0], dtype=torch.float32, requires_grad=False)
target_eigenvalues_napthalene = torch.tensor([-13.0, -12.0, -11.5, -12.5, -11.0, -10.5, -10.0, -9.0, -9.5, -8.0], dtype=torch.float32, requires_grad=False)

benzene_coordinates = np.array([
    [-4.461121, 1.187057, -0.028519],
    [-3.066650, 1.263428, -0.002700],
    [-2.303848, 0.094131, 0.041626],
    [-2.935547, -1.151550, 0.059845],
    [-4.330048, -1.227982, 0.034073],
    [-5.092743, -0.058655, -0.010193]
])

napthalene_coordinates = np.array([
    [ 1.24593, 1.40391, -0.0000],
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

# create instance for benzene
molecular_system_benzene = MolecularSystem(benzene_coordinates, alpha_initial, beta_initial, cutoff_distance)
optimise_and_plot(molecular_system_benzene, target_eigenvalues_benzene, "Benzene", cutoff_distance)

# create instance for napthalene
#molecular_system_napthalene = MolecularSystem(napthalene_coordinates, alpha_initial, beta_initial, cutoff_distance)
#optimise_and_plot(molecular_system_napthalene, target_eigenvalues_napthalene, "Naphthalene", cutoff_distance)






