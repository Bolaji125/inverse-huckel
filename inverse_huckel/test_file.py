import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Create a 2x2 matrix
matrix = torch.tensor([[3.0, 1.0],
                       [1.0, 4.0]], requires_grad=True)

# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = torch.linalg.eig(matrix)
  
# Extract the real parts of the eigenvalues
eigenvalues_real = eigenvalues.real
print(eigenvalues_real)

# target eigenvalues
target_eigenvalues_real = torch.tensor([0.5, 1.5]) # These are the values that we want the eigenvalues of our matrix to be close to

# Compute the MSE loss for the real parts
loss = F.mse_loss(eigenvalues_real, target_eigenvalues_real) # This computes the mean squared error (MSE) loss between the computed eigenvalues and the target eigenvalues

# Backpropagate
loss.backward() # backpropagates through the loss to compute gradients of the parameters (matrix in this case) with respect to the los

# Collect gradients of the elements with respect to the eigenvalues
gradients = matrix.grad

print("Gradients with respect to the elements of the matrix:", gradients)

# Plot the original matrix
plt.figure(figsize=(20,3))
plt.subplot(1, 4, 1)
plt.imshow(matrix.detach().numpy(), cmap='viridis', interpolation='nearest')
plt.xlabel('Column')
plt.ylabel('Row')
plt.title('Visualisation of Matrix')
plt.colorbar()


# Plot the gradients

# plt.subplot(1, 5, 2)
# plt.imshow(gradients.detach().numpy(), cmap='viridis', interpolation='nearest')
# plt.xlabel('Column')
# plt.ylabel('Row')
# plt.title('Visualisation of Gradients')
# plt.colorbar()


# Plot the derivative of the matrix with respect to each eigenvalue
if eigenvalues.dim() == 0:
    print("Eigenvalues tensor is 0-dimensional. Unable to plot derivatives.")
else:
    for i in range(len(eigenvalues)):
        try:
            plt.subplot(1, 4, 2 + i)
            #matrix_grad = torch.outer(eigenvectors[:, i].real, eigenvectors[:, i].real)
            matrix_derivative = eigenvectors[:, i].real.unsqueeze(1) @ eigenvectors[:, i].real.unsqueeze(0)
            print(f'Derivative of Matrix wrt Eigenvalue {i+1}', matrix_derivative)
            #plt.imshow(matrix_grad.detach().numpy(), cmap='viridis', interpolation='nearest')
            plt.imshow(matrix_derivative.detach().numpy(), cmap='viridis', interpolation='nearest')
            plt.xlabel('Column')
            plt.ylabel('Row')
            plt.title(f'Derivative of Matrix wrt Eigenvalue {i+1}')
            plt.colorbar()
            
        except Exception as e:
            print(f"Error plotting derivative for eigenvalue {i+1}: {e}")

# Update the matrix using gradients
learning_rate = 0.1 # determines size of the step taken during the update process
with torch.no_grad():
    matrix -= learning_rate * gradients
    print("new matrix values:", matrix)

# Compute the eigenvalues and eigenvectors of the matric after the update
eigenvalues_after, eigenvectors_after = torch.linalg.eig(matrix)

# Extract the real parts of the eigenvalues after the update
eigenvalues_real_after = eigenvalues_after.real
print("eigenavlues after", eigenvalues_real_after)

# Define a new loss function: subtraction
def minus_eigenvalue_loss(eigenvalues_real, target_eigenvalue_real):
    return (eigenvalues_real[0] - target_eigenvalue_real)**2  # Subtract the target eigenvalue from the first eigenvalue

# Compute the new loss
new_loss = minus_eigenvalue_loss(eigenvalues_real_after, target_eigenvalues_real[0])

# Backpropagate through the new loss
matrix.grad = None  # Clear previous gradients
new_loss.backward()

# Compute the derivative of the matrix after backpropagation with respect to the new eigenvalue
derivative_matrix_wrt_eigenvalue = matrix.grad

print("Derivative of the matrix after backpropagation with respect to the new eigenvalue:")
print(derivative_matrix_wrt_eigenvalue)

# Plot the matrix after the update
plt.subplot(1, 4, 4)
plt.imshow(matrix.detach().numpy(), cmap='viridis', interpolation='nearest')
plt.xlabel('Column')
plt.ylabel('Row')
plt.title('Derivative of the matrix after backpropagation wrt new eigenvalue 1')
plt.colorbar()
plt.tight_layout()
plt.show()
    

