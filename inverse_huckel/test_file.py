import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Create a 2x2 matrix
matrix = torch.tensor([[1.0, 2.0],
                       [3.0, 4.0]], requires_grad=True)

# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = torch.linalg.eig(matrix)
  

# Extract the real parts of the eigenvalues
eigenvalues_real = eigenvalues.real

# Extract the real parts of the target eigenvalues
target_eigenvalues_real = torch.tensor([0.5, 1.5]) # These are the values that we want the eigenvalues of our matrix to be close to

# Compute the MSE loss for the real parts
loss = F.mse_loss(eigenvalues_real, target_eigenvalues_real) # This computes the mean squared error (MSE) loss between the computed eigenvalues and the target eigenvalues

# Backpropagate
loss.backward() # backpropagates through the loss to compute gradients of the parameters (matrix in this case) with respect to the los

# Collect gradients of the elements with respect to the eigenvalues
gradients = matrix.grad

print("Gradients with respect to the elements of the matrix:", gradients)

# Plot the original matrix
plt.figure()
plt.imshow(matrix.detach().numpy(), cmap='viridis', interpolation='nearest')
plt.xlabel('Column')
plt.ylabel('Row')
plt.title('Visualisation of Matrix')
plt.colorbar()
plt.show()

# Plot the gradients
plt.figure()
plt.imshow(gradients.detach().numpy(), cmap='viridis', interpolation='nearest')
plt.xlabel('Column')
plt.ylabel('Row')
plt.title('Visualisation of Gradients')
plt.colorbar()
plt.show()

# Plot the derivative of the matrix with respect to each eigenvalue
if eigenvalues.dim() == 0:
    print("Eigenvalues tensor is 0-dimensional. Unable to plot derivatives.")
else:
    for i in range(len(eigenvalues)):
        try:
            plt.figure()
            matrix_grad = torch.outer(eigenvectors[:, i].real, eigenvectors[:, i].real)
            plt.imshow(matrix_grad.detach().numpy(), cmap='viridis', interpolation='nearest')
            plt.xlabel('Column')
            plt.ylabel('Row')
            plt.title(f'Derivative of Matrix wrt Eigenvalue {i+1}')
            plt.colorbar()
            plt.show()
        except Exception as e:
            print(f"Error plotting derivative for eigenvalue {i+1}: {e}")

        # plt.figure()
        # matrix_grad = torch.outer(eigenvectors[:, i].real, eigenvectors[:, i].real) # torch.outer() computes the outer product of the i-th eigenvector with itself.constructs a matrix whose elements represent the derivative of the matrix with respect to the i-th eigenvalue
        # plt.imshow(matrix_grad.detach().numpy(), cmap='viridis', interpolation='nearest')
        # plt.xlabel('Column')
        # plt.ylabel('Row')
        # plt.title(f'Derivative of Matrix wrt Eigenvalue {i+1}')
        # plt.colorbar()
        # plt.show()
    

