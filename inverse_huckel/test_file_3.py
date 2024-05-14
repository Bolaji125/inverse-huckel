import torch
import torch.nn.functional as F

# Create a 2x2 matrix
matrix = torch.tensor([[3.0, 1.0],
                       [2.0, 4.0]], requires_grad=True)

# target eigenvalues
target_eigenvalues_real = torch.tensor([0.5, 1.5])

# Learning rate
learning_rate = 0.1

# Optimization loop
for i in range(5):  
    # Compute the eigenvalues
    eigenvalues, _ = torch.linalg.eig(matrix)
    print(matrix)
  
    # Extract the real parts of the eigenvalues
    eigenvalues_real = eigenvalues.real

    # Compute the MSE loss for the real parts
    loss = F.mse_loss(eigenvalues_real, target_eigenvalues_real)

    # Print the loss value at each iteration
    print(f"Iteration {i+1}, Loss: {loss.item()}")

    # Backpropagate
    loss.backward()

    # Manually update the specific value in the matrix
    # with torch.no_grad():
    #     matrix[0, 0] -= learning_rate * matrix.grad[0, 0]

    # Manually update the diagonal values in the matrix
    # with torch.no_grad():
    #     for j in range(matrix.shape[0]):  # Loop over the diagonal elements
    #         matrix[j, j] -= learning_rate * matrix.grad[j, j]

    # Manually update the off-diagonal elements in the matrix
    with torch.no_grad():
        for j in range(matrix.shape[0]):  # Loop over rows
            for k in range(matrix.shape[1]):  # Loop over columns
                if j != k:  # If not on the diagonal
                    matrix[j, k] -= learning_rate * matrix.grad[j, k]


    # Zero gradients
    matrix.grad.zero_()

# Print the optimised matrix
print("Optimised Matrix:", matrix)

# Compute the eigenvalues after optimization
# eigenvalues_after, _ = torch.linalg.eig(matrix)
# print("Eigenvalues after optimization:", eigenvalues_after.real)
print("eigenvalues after optimisation", eigenvalues_real)