import numpy as np
from matplotlib import pyplot as plt

# Constants
mass = 1.0
hbar = 1.0

# Define grid
num_grid = 64
length = 20.0
dx = length / (num_grid + 1)
xj = np.linspace(-length / 2 + dx, length / 2 - dx, num_grid)

# Hamiltonian Matrix
ham_mat = np.zeros((num_grid,num_grid))

for i in range(num_grid):
    for j in range(num_grid):
        if(i == j):
            ham_mat[i,j]=-0.5*hbar**2/mass*(-2.0/dx**2)
        elif(np.abs(i-j) == 1):
            ham_mat[i,j]=-0.5*hbar**2/mass*(1.0/dx**2)    

# Calculate eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eigh(ham_mat)

# Normalize and check the sign
wf = eigenvectors/np.sqrt(dx)
for i in range(num_grid):
    sign = np.sign(wf[num_grid//2,i])
    if(sign != 0.0):
        wf[:,i] = wf[:,i]*sign


def exact_eigenvalue(n):
    """Calculate exact eigenvalue for particle in a box."""
    return n**2 * np.pi**2 * hbar**2 / (2.0 * mass * length**2)

# Print eigenvalues and errors
for i in range(3):
    print(f"{i}-th eigenvalue = {eigenvalues[i]}")
    print(f"{i}-th eigenvalue Error = {eigenvalues[i] - exact_eigenvalue(i + 1)}")
    print()

# Plotting
plt.figure(figsize=(8,6))
plt.plot(xj, wf[:, 0], label="Ground state (calc.)")
plt.plot(xj, np.sqrt(2.0/length)*np.cos(np.pi*xj/length), label="Ground state (exact.)", linestyle='dashed')
plt.plot(xj, wf[:, 1], label="1st excited state")
plt.plot(xj, np.sqrt(2.0/length)*np.sin(2.0*np.pi*xj/length), label="1st excited state (exact.)", linestyle='dashed')
plt.plot(xj, wf[:, 2], label="2nd excited state")
plt.plot(xj, np.sqrt(2.0/length)*np.cos(3.0*np.pi*xj/length), label="2nd excited state (exact.)", linestyle='dashed')

plt.xlim([-length/2.0,length/2.0])
plt.xlabel('x')
plt.ylabel('wave functions')
plt.legend()
plt.savefig('fig_quantum_well_wf.pdf')
plt.show()
