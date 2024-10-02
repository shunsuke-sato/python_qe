import numpy as np
from matplotlib import pyplot as plt

# define grid
num_grid = 256
radius = 20.0
dr = radius/num_grid

rj = np.linspace(dr, radius-dr, num_grid)


# Hamiltnian matrix
ham_mat = np.zeros((num_grid, num_grid))


for i in range(num_grid):
    for j in range(num_grid):
        if(i == j):
            ham_mat[i,j] = -0.5*(-2.0/dr**2) - 1.0/rj[j]
        elif(np.abs(i-j) == 1):
            ham_mat[i,j] = -0.5*(1.0/dr**2)


# Calculate eigenvectors and eigenvalues
eigenvalues , eigenvectors = np.linalg.eigh(ham_mat)


# Print the eigenvalues
print('Energy (1s)=', eigenvalues[0])
print('Energy (2s)=', eigenvalues[1])
print('Energy (3s)=', eigenvalues[2])


# Calculate wavefunctions and coordinate
rj_new = np.linspace(0.0, radius-dr, num_grid+1)
wf = np.zeros((num_grid+1, num_grid))

for iorb in range(num_grid):
    wf[1:num_grid,iorb] = eigenvectors[0:num_grid-1,iorb]/(4.0*np.pi*dr*rj[0:num_grid-1])

wf[0,:] = 2.0*wf[1,:]-wf[2,:]

plt.title("Probability, $r^2*|\Psi(r)|^2$")
plt.plot(rj_new, rj_new[:]**2*wf[:,0]**2, label="Ground state (1s)")
plt.plot(rj_new, rj_new[:]**2*wf[:,1]**2, label="1st-excited state (2s)")
plt.plot(rj_new, rj_new[:]**2*wf[:,2]**2, label="2nd-excited state (3s)")
plt.xlim(0.0,15)
plt.xlabel("Radius (Bohr)")
plt.ylabel("Radial probability distribution")
plt.legend()

plt.savefig('hydrogen_wf.pdf')
plt.show()

