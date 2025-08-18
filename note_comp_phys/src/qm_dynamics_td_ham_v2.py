from numba import jit
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter


# Construct potential
def construct_potential(xj, xc):
    return 0.5*(xj-xc)**2

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

# Operate the Hamiltonian to the wavefunction
@jit(nopython=True)
def ham_wf(wf, potential, dx):

    n = wf.size
    hwf = np.zeros(n, dtype=np.complex128)
    
    for i in range(1,n-1):
        hwf[i] = -0.5*(wf[i+1]-2.0*wf[i]+wf[i-1])/(dx**2)

    i = 0
    hwf[i] = -0.5*(wf[i+1]-2.0*wf[i])/(dx**2)
    i = n-1
    hwf[i] = -0.5*(-2.0*wf[i]+wf[i-1])/(dx**2)
    
    hwf = hwf + potential*wf
    
    return hwf


# Time propagation from t to t+dt
def time_propagation(wf, potential, dx, dt):

    n = wf.size
    twf = np.zeros(n, dtype=complex)
    hwf = np.zeros(n, dtype=complex)

    twf = wf
    zfact = 1.0 + 0j
    for iexp in range(1,5):
        zfact = zfact*(-1j*dt)/iexp
        hwf = ham_wf(twf, potential, dx)
        wf = wf + zfact*hwf
        twf = hwf

    return wf


# time propagation parameters
#omega = 8.0
omega = 0.2
Tprop = 80.0
dt = 0.005
nt = int(Tprop/dt)+1

# set the coordinate
xmin = -10.0
xmax = 10.0
num_grid = 250

xj = np.linspace(xmin, xmax, num_grid)
dx = xj[1] - xj[0]

# set potential
xc = 1.0
potential = construct_potential(xj, xc)

# calculate the ground state
wf = calc_ground_state(xj, potential)


# For loop for the time propagation
density_list = []
xc_list = []
for it in range(nt+1):
    tt = it*dt
    xc = np.cos(omega*tt)
    if(it % (nt//200) == 0):
        rho = np.abs(wf)**2
        density_list.append(rho.copy())
        xc_list.append(xc)


    potential = construct_potential(xj, xc)

    wf = time_propagation(wf, potential, dx, dt)
    print(it, nt)

# plot the density, |wf|^2

# Define function to update plot for each frame of the animation
def update_plot(frame):
    plt.cla()
    plt.xlim([-5.0, 5.0])
    plt.ylim([0.0, 0.8])
    xc = xc_list[frame]
    plt.plot(xj, density_list[frame], label="$|\psi(x)|^2$ (calc.)")
    plt.plot(xj, np.exp(-(xj-xc)**2)/np.sqrt(np.pi), 
             label="$|\psi(x)|^2$ (ref.)", linestyle='dashed')
    plt.plot(xj, 0.5*(xj-xc)**2, 
         label="Harmonic potential", linestyle='dotted')

    plt.xlabel('x')
    plt.ylabel('Density, Potential')
    plt.legend(loc = 'upper right')


# Create the animation
fig = plt.figure()
ani = animation.FuncAnimation(fig, update_plot, frames=len(density_list), interval=50)
#ani.save('wavefunction_animation.gif', writer='imagemagick')
ani.save('density_animation.gif', writer='pillow')

