import numpy as np

matrix = np.array([[0.0, 1.0, 0.0],
                   [1.0, 0.0, 1.0],
                   [0.0, 1.0, 0.0]])

eigenvalues, eigenvectors = np.linalg.eigh(matrix)

print('First eigenvalue  =',eigenvalues[0])
print('Second eigenvalue =',eigenvalues[1])
print('Third eigenvalue  =',eigenvalues[2])
print()

print('First eigenvector')
print(eigenvectors[:,0])
print()

print('Second eigenvector')
print(eigenvectors[:,1])
print()

print('Third eigenvector')
print(eigenvectors[:,2])
print()
