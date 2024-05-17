import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

class MolecularSystem:

    def __init__(self, coordinates, alpha, beta, cutoff_distance):
        self.coordinates = torch.tensor(coordinates, dtype=torch.float32)
        self.alpha = torch.tensor([alpha] * len(coordinates), requires_grad=True) # explain syntax again
        self.beta = torch.tensor([beta] * len(coordinates), requires_grad=True)
        self.cutoff_distance = cutoff_distance
        self.H = torch.zeros((len(coordinates), len(coordinates)), dtype=torch.float32)
        self.update_hamiltonian() # calls update hamiltonian to intialise the matrix, what happened to construct hamiltonian????

    def calculate_distance(self, atom_i, atom_j): 
        atom_i_tensor = torch.tensor(atom_i)
        atom_j_tensor = torch.tensor(atom_j)
        return torch.linalg.norm(atom_i_tensor - atom_j_tensor)

    def update_hamiltonian(self): #updates hamiltonian based on the current values of alpha, beta, and the cutoff distance
        num_atoms = self.coordinates.shape[0]
        self.H = torch.zeros((num_atoms, num_atoms), dtype=torch.float32)
       
        # Set diagonal elements to alpha values
        for i in range(num_atoms):
            self.H[i, i] = self.alpha[i]

        # Set off-diagonal elements based on cut off distance and beta
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    distance = torch.norm(self.coordinates[i] - self.coordinates[j])
                    if distance <= self.cutoff_distance:
                        self.H[i, j] = self.beta[i]
                        self.H[j, i] = self.beta[i]

    def solve_eigenvalue_problem_pytorch(self):
        eigenvalues, eigenvectors = torch.linalg.eigh(self.H)
        return eigenvalues, eigenvectors
    
def plot_molecule(self, molecule_name="Molecule"):
    fig, ax = plt.subplots()
    num_atoms = len(self.coordinates)

    # Plot atoms with sizes based on alpha values
    for i in range(num_atoms):
        ax.scatter(self.coordinates[i, 0], self.coordinates[i, 1], s=self.alpha[i].item() * 1000, label=f'Atom {i+1}')
    
    # Plot bonds with widths based on beta values
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = torch.norm(self.coordinates[i] - self.coordinates[j])
            if distance <= self.cutoff_distance:
                ax.plot([self.coordinates[i, 0], self.coordinates[j, 0]], [self.coordinates[i, 1], self.coordinates[j, 1]], 
                        'k-', linewidth=self.beta[i].item() * 10)

    ax.set_xlabel('X / $\AA$')
    ax.set_ylabel('Y / $\AA$')
    ax.set_title(f'{molecule_name} Structure')
    ax.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()



def optimise_molecular_system(molecular_system, target_eigenvalues, num_iterations=1000, learning_rate=0.01): # Performs gradient descent to optimise the alpha and beta parameters to match the target eigenvalues.
    alpha_history = [] #initialises an empty list to store the history of alpha values at each iteration
    beta_history = []
    loss_history = []
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

def plot_parameter_changes(alpha_history, beta_history, loss_history,molecule_name): #plots the changes in lpha and beta over the optmisation epochs
    epochs = np.arange(alpha_history.shape[0]) # determined by the number of rows in the alpha_history array (assuming alpha_history and beta_history have the same number of epochs).each row typically represents one training iteration or epoch.
    print(f'Alpha history for {molecule_name}:', alpha_history)
    print(f'Beta history for {molecule_name}:', beta_history)
    print(f"Number of epochs in alpha history for {molecule_name}: {alpha_history.shape[0]}")
    print(f"Number of epochs in beta history for {molecule_name}: {beta_history.shape[0]}")
    
    # creating subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 18)) # creates a figure with 2 subplots arranged vertically (2 rows, 1 column) and sets the figure size to (12, 14) inches
    # plotting alpha changes
    for i in range(alpha_history.shape[1]): #loop iterates over the column as each column correspons to a different atom index
        axs[0].plot(epochs, alpha_history[:, i], label=f'Alpha Atom {i+1}') # plots the alpha values for the current atom index against epochs on the first subplot.
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Alpha Value')
    axs[0].set_title(f'Alpha Parameter Changes for {molecule_name}')
    axs[0].legend()
    axs[0].grid(True)

    # plotting beta changes
    for i in range(beta_history.shape[1]):
        axs[1].plot(epochs, beta_history[:, i], label=f'Beta Atom {i+1}')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Beta Value')
    axs[1].set_title(f'Beta Parameter Changes for {molecule_name}')
    axs[1].legend()
    axs[1].grid(True)

    # Plotting loss changes
    axs[2].plot(epochs, loss_history, label='Loss', color='red')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Loss')
    axs[2].set_title(f'Loss Changes for {molecule_name}')
    axs[2].legend()
    axs[2].grid(True)

    #plt.tight_layout()
    plt.subplots_adjust(hspace=0.6)  # Add space between subplots
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

# create instance for each molecule
molecular_system_benzene = MolecularSystem(benzene_coordinates, alpha_initial, beta_initial, cutoff_distance)
molecular_system_napthalene = MolecularSystem(napthalene_coordinates, alpha_initial, beta_initial, cutoff_distance)

# optimise the molecular system
alpha_history_benzene, beta_history_benzene,loss_history_benzene = optimise_molecular_system(molecular_system_benzene, target_eigenvalues_benzene)
alpha_history_napthalene, beta_history_napthalene, loss_history_napthalene = optimise_molecular_system(molecular_system_napthalene, target_eigenvalues_napthalene)

# plot the changes
plot_parameter_changes(alpha_history_benzene, beta_history_benzene, loss_history_benzene, "Benzene")
plot_molecule(molecular_system_benzene, "Benzene")

plot_parameter_changes(alpha_history_napthalene, beta_history_napthalene, loss_history_napthalene, "Naphthalene")
plot_molecule(molecular_system_napthalene, "Naphthalene")