import torch
import torch.nn.functional as F

# Define your initial Hamiltonian matrix
initial_hamiltonian = torch.tensor([[1.0, 0.5, 0.2],
                                    [0.5, 2.0, 0.3],
                                    [0.2, 0.3, 3.0]], requires_grad=True)

# Target eigenvalues
target_eigenvalues = torch.tensor([0.5, 1.5, 2.5])

# Learning rate
learning_rate = 0.1

# Optimization loop
for i in range(100):
    # Compute the eigenvalues of the current Hamiltonian
    eigenvalues, _ = torch.linalg.eig(initial_hamiltonian)
  
    # Compute the MSE loss for the eigenvalues, excluding the diagonal elements
    off_diagonal_indices = torch.where(~torch.eye(len(eigenvalues), dtype=bool))
    loss = F.mse_loss(eigenvalues[off_diagonal_indices[0]], target_eigenvalues)

    # Print the loss value at each iteration
    print(f"Iteration {i+1}, Loss: {loss.item()}")

    # Backpropagate
    loss.backward()

    # Manually update the off-diagonal elements of the Hamiltonian matrix
    with torch.no_grad():
        off_diagonal_grad = initial_hamiltonian.grad.clone()  # Copy the gradient tensor
        off_diagonal_grad[torch.eye(len(initial_hamiltonian), dtype=bool)] = 0  # Zero out the gradients for diagonal elements
        initial_hamiltonian -= learning_rate * off_diagonal_grad

    # Zero gradients
    initial_hamiltonian.grad.zero_()

# Print the optimized Hamiltonian matrix
print("Optimized Hamiltonian Matrix:")
print(initial_hamiltonian)

# Compute the eigenvalues after optimization
eigenvalues_after, _ = torch.linalg.eig(initial_hamiltonian)
print("Eigenvalues after optimization:")
print(eigenvalues_after.real)