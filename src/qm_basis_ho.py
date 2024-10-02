import numpy as np
from matplotlib import pyplot as plt

# Constants
mass = 1.0
hbar = 1.0
kconst = 1.0
omega = np.sqrt(kconst/mass)
length = 15.0

num_basis = 16

ham_mat = np.zeros((num_basis, num_basis))

for m in range(num_basis):
    for n in range(num_basis):
        mb = m + 1
        nb = n + 1
        
        if(mb == nb):
            ham_mat[m,n]=length**2*kconst*((1/24.0-1/(4.0*nb**2*np.pi**2)))+0.5*hbar**2/mass*(nb*np.pi/length)**2
        elif(mb%2 == nb%2):
            ham_mat[m,n]=length**2*kconst*(-1)**(mb%2+(mb+nb)//2)*4*mb*nb/((mb-nb)**2*(mb+nb)**2*np.pi**2)


# Define grid
num_grid = 512
dx = length / (num_grid + 1)
xj = np.linspace(-length / 2 + dx, length / 2 - dx, num_grid)


# Calculate eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eigh(ham_mat)

wf = np.zeros((num_grid, num_basis))

for m in range(num_basis):
    for n in range(num_basis):
        nb = n + 1
        if(nb%2 == 0):
            wf[:,m] = wf[:,m] + np.sqrt(2.0/length)*np.sin(nb*np.pi*xj[:]/length)*eigenvectors[n,m]
        else:
            wf[:,m] = wf[:,m] + np.sqrt(2.0/length)*np.cos(nb*np.pi*xj[:]/length)*eigenvectors[n,m]
            


def exact_eigenvalue(n):
    return hbar * np.sqrt(kconst / mass) * (n + 0.5)

# Print eigenvalues and errors
for i in range(3):
    print(f"{i}-th eigenvalue = {eigenvalues[i]}")
    print(f"{i}-th eigenvalue Error = {eigenvalues[i] - exact_eigenvalue(i )}")
    print()

# Plotting
plt.plot(xj, wf[:, 0], label="Ground state wf (calc.)")
plt.plot(xj, (mass*omega/(hbar*np.pi))**0.25*np.exp(-0.5*mass*omega*xj**2/hbar), label="Ground state wf (exact)", linestyle="dashed")
plt.xlabel('x')
plt.ylabel('wavefunction')
plt.legend()
plt.savefig('fig_wf_ho_well_basis.jpeg')
plt.show()
