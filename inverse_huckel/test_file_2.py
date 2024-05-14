import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Create a 2x2 matrix
matrix = torch.tensor([[3.0, 1.0],
                       [1.0, 4.0]], requires_grad=True)

# target eigenvalues
target_eigenvalues_real = torch.tensor([0.5, 1.5]) # These are the values that we want the eigenvalues of our matrix to be close to

# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = torch.linalg.eig(matrix)
  
    # Extract the real parts of the eigenvalues
eigenvalues_real = eigenvalues.real
print("eigenvalues before optimisation", eigenvalues_real)

# Create an optimizer
#optimizer = torch.optim.SGD([matrix], lr=0.1) #initialises the stochastic gradient descent (SGD) optimizer, which will be used to optimize the matrix parameters (matrix) with a learning rate of 0.1

learning_rate = 0.1

# Optimization loop
for i in range(5):  # Backpropagate many times
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eig(matrix)
    print(matrix)
  
    # Extract the real parts of the eigenvalues
    eigenvalues_real = eigenvalues.real

    # Compute the MSE loss for the real parts
    loss = F.mse_loss(eigenvalues_real, target_eigenvalues_real)

    # Print the loss value at each iteration
    print(f"Iteration {i+1}, Loss: {loss.item()}")

    # Zero gradients
    #optimizer.zero_grad() #Before performing backpropagation, we need to zero out the gradients stored in the optimizer.

    # Backpropagate
    loss.backward()

    gradients = matrix.grad
    matrix = matrix.detach() - learning_rate * gradients
    # Update the matrix using the optimizer
    #optimizer.step()

# Print the optimized matrix
print("Optimized Matrix:", matrix)

print("eigenvalues after optimisation", eigenvalues_real)