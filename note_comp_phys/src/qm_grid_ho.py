import numpy as np
from matplotlib import pyplot as plt

# Constants
mass = 1.0
hbar = 1.0
kconst = 1.0

# Define grid
num_grid = 128
length = 15.0
dx = length / (num_grid + 1)
xj = np.linspace(-length / 2 + dx, length / 2 - dx, num_grid)

# Potential
vpot = 0.5*kconst*xj**2

# Hamiltonian Matrix
ham_mat = np.zeros((num_grid,num_grid))

for i in range(num_grid):
    for j in range(num_grid):
        if(i == j):
            ham_mat[i,j]=-0.5*hbar**2/mass*(-2.0/dx**2) + vpot[i]
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
    """Calculate exact eigenvalue for quantum harmonic oscillator."""
    return hbar * np.sqrt(kconst / mass) * (n + 0.5)

# Print eigenvalues and errors
for i in range(3):
    print(f"{i}-th eigenvalue = {eigenvalues[i]}")
    print(f"{i}-th eigenvalue Error = {eigenvalues[i] - exact_eigenvalue(i)}")
    print()


# Plotting
omega = np.sqrt(kconst/mass)

# Ground state plot
plt.figure(figsize=(8, 6))
plt.plot(xj, wf[:, 0], label="Ground state (calc.)")
plt.plot(xj, (mass * omega / (np.pi * hbar))**(1.0 / 4.0) * np.exp(-mass * omega * xj**2 / (2.0 * hbar)),
                  label="Ground state (exact.)", linestyle='dashed')
plt.xlim([-length / 2.0, length / 2.0])
plt.xlabel('x')
plt.ylabel('wave functions')
plt.legend()
plt.savefig('fig_harmonic_oscillator_ground_state.pdf')
plt.show()

# First excited state plot
plt.figure(figsize=(8, 6))
plt.plot(xj, wf[:, 1], label="1st excited state (calc.)")
plt.plot(xj, (mass * omega / (np.pi * hbar))**(1.0 / 4.0) * np.sqrt(2.0 * mass * omega / hbar) * xj * np.exp(-mass * omega * xj**2 / (2.0 * hbar)),
                  label="1st excited state (exact.)", linestyle='dashed')
plt.xlim([-length / 2.0, length / 2.0])
plt.xlabel('x')
plt.ylabel('wave functions')
plt.legend()
plt.savefig('fig_harmonic_oscillator_1st_excited_state.pdf')
plt.show()

# Second excited state plot
plt.figure(figsize=(8, 6))
plt.plot(xj, wf[:, 2], label="2nd excited state (calc.)")
plt.plot(xj, (mass * omega / (np.pi * hbar))**(1.0 / 4.0) * np.sqrt(0.5) * (1.0 - 2.0 * mass * omega * xj**2 / hbar) * np.exp(-mass * omega * xj**2 / (2.0 * hbar)),
                  label="2nd excited state (exact.)", linestyle='dashed')
plt.xlim([-length / 2.0, length / 2.0])
plt.xlabel('x')
plt.ylabel('wave functions')
plt.legend()
plt.savefig('fig_harmonic_oscillator_2nd_excited_state.pdf')
plt.show()
