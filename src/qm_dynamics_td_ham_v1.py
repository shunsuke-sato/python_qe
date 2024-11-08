from numba import jit
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter


# Construct potential
def construct_potential(xj, xc):
    return 0.5*k0*(xj-xc)**2

def calc_ground_state(xj, potential):

    num_grid = xj.size
    dx = xj[1]-xj[0]

    ham = np.zeros((num_grid, num_grid))

    for i in range(num_grid):
        for j in range(num_grid):
            if(i == j):
                ham[i,j] = -0.5*(-2.0/dx**2)+potential[i]
            elif(np.abs(i-j)==1):
                ham[i,j] = -0.5*(1.0/dx**2)

    eigenvalues, eigenvectors = np.linalg.eigh(ham)

    wf = np.zeros(num_grid, dtype=complex)

    wf.real = eigenvectors[:,0]/np.sqrt(dx)

    return wf

# time propagation parameters
Tprop = 40.0
dt = 0.005
#dt = 0.00905
nt = int(Tprop/dt)+1

# set the coordinate
xmin = -10.0
xmax = 10.0
num_grid = 250

xj = np.linspace(xmin, xmax, num_grid)

# set potential
xc = -1.0
potential = construct_potential(xj, xc)

# calculate the ground state
wf = calc_ground_state(xj, potential)



# plot the ground state density, |wf|^2
rho = np.abs(wf)**2

plt.figure(figsize=(8,6))
plt.plot(xj, rho, label="$|\psi(x)|^2$ (calc.)")
plt.plot(xj, np.exp(-(xj-xc)**2)/np.sqrt(np.pi), 
         label="$|\psi(x)|^2$ (exact)", linestyle='dashed')
plt.plot(xj, 0.5*(xj-xc)**2, 
         label="Harmonic potential", linestyle='dotted')

plt.xlim([-4.0, 4.0])
plt.ylim([0.0, 0.8])
plt.xlabel('x')
plt.ylabel('Density, Potential')
plt.legend()
plt.savefig('gs_density.pdf')
plt.show()



