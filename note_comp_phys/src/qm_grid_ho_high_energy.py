import numpy as np
from matplotlib import pyplot as plt

# Constants
mass = 1.0
hbar = 1.0
kconst = 1.0

# Define grid
num_grid = 512
length = 30.0
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


# Calculate the probability distribution fo a highly-excited state
n_eigen = 64
Ene = eigenvalues[n_eigen]

prob_x = np.zeros(num_grid)
for i in range(num_grid):
    if( 2.0*Ene/kconst-xj[i]**2 < 0):
        prob_x[i]=0.0
    else:
        prob_x[i]=1.0/(np.pi*np.sqrt(2.0*Ene/kconst-xj[i]**2))

plt.figure(figsize=(8,6))

plt.plot(xj, wf[:, n_eigen]**2, label="Quantum distribution")
plt.plot(xj, prob_x, label="Classical distribution")
plt.xlim([-length/2.0,length/2.0])
plt.xlabel('x')
plt.ylabel('$|\psi(x)|^2$')
plt.legend()
plt.savefig('harmonic_oscillator_high_energy_prob.pdf')
plt.show()




