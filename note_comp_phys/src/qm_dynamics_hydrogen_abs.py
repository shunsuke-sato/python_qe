import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

# define the potential
def calc_potential(xj):
    vpot = -1.0/np.sqrt(xj**2+1.0)
    return vpot


def calc_static_hamiltonian(num_grid, xj, vpot):

    ham = np.zeros((num_grid, num_grid))
    dx = xj[1]-xj[0]

    for i in range(num_grid):
        for j in range(num_grid):
            if(i == j):
                ham[i,j] = -0.5*(-2.0/dx**2)+vpot[i]
            elif(np.abs(i-j)==1):
                ham[i,j] = -0.5*(1.0/dx**2)


    return ham
            

def calc_gs_wf(num_grid, ham):
    
    eigenvalues, eigenvectors = np.linalg.eigh(ham)
    print("gs energy =", eigenvalues[0])

    wf = np.zeros(num_grid, dtype=complex)
    wf.real = eigenvectors[:,0]
    wf[0] = 0.0
    wf[-1] = 0.0

    return wf


def calc_laser_field(tt):

    omega0 = 0.05
    E0 = 0.1
    tpulse = 10*2.0*np.pi/omega0
    xx = tt-0.5*tpulse

    if(np.abs(xx)/tpulse < 0.5):
        Et = E0*np.cos(np.pi*xx/tpulse)**2*np.sin(omega0*xx)
    else:
        Et = 0.0

    return Et



def ham_wf(wf, vpot, dx):

    num_grid = wf.size
    hwf = np.zeros(num_grid, dtype=complex)

    for i in range(1, num_grid-1):
        hwf[i] = -0.5*(wf[i+1]-2.0*wf[i]+wf[i-1])/(dx**2)

    hwf = hwf + vpot*wf
    return hwf
    

def time_propagation(xj, wf, vpot, dx, tt, dt):
    
    num_grid = xj.size

    Et = calc_laser_field(tt)
    v_Et = -Et*xj

    # set absorbing boundary
    v_abs = np.zeros(num_grid, dtype=complex)
    for i in range(num_grid):
        if(np.abs(xj[i]) > 40.0):
            v_abs[i] = -0.2*1j*(np.abs(xj[i]) - 40.0)

    

    # apply exp(-0.5*0j*v_Et*dt)
    wf = wf*np.exp(-0.5*1j*(v_Et+v_abs)*dt)

    # propagate 
    twf = wf
    

    zfact = 1.0 + 0j
    for iexp in range(1,5):
        zfact = zfact*(-1j*dt)/iexp
        hwf = ham_wf(twf, vpot, dx)
        wf = wf + zfact*hwf
        twf = hwf


    # apply exp(-0.5*0j*v_Et*dt)
    wf = wf*np.exp(-0.5*1j*(v_Et+v_abs)*dt)

    return wf

def calc_dipole(xj, wf):
    
    dx = xj[1]-xj[0]
    dipole = np.sum(np.abs(wf)**2*xj)*dx

    return dipole


# Set the coordinate
xmin = -50.0
xmax = 50.0
num_grid = 500

xj = np.linspace(xmin, xmax, num_grid)
dx = xj[1]-xj[0]

# Time propagation parameters
Tprop = 1300.0 #80.0
dt = 0.05
nt = int(Tprop/dt)+1




# set the potential
vpot = calc_potential(xj)

# set the static Hamiltonian
ham = calc_static_hamiltonian(num_grid, xj, vpot)

# set the initial wavefunction (ground state)
wf = calc_gs_wf(num_grid, ham)

# set output quantities
tt_out = np.zeros(nt)
Et_out = np.zeros(nt)
dipole_out = np.zeros(nt)
norm_out = np.zeros(nt)

# wavefunction array to make a movie
wavefunctions = []

# For loop for the time propagation
for it in range(nt):
    if(it%(nt//100) == 0):
        print("it=",it,nt)
        wavefunctions.append(wf.copy())

    tt = dt*it

    # compute outputs
    tt_out[it] = dt*it
    Et_out[it] = calc_laser_field(tt)
    dipole_out[it] = calc_dipole(xj, wf)
    norm_out[it] = np.sum(np.abs(wf)**2)*dx

    wf = time_propagation(xj, wf, vpot, dx, tt, dt)


plt.figure()
plt.plot(tt_out, Et_out, label="E(t)")
plt.plot(tt_out, dipole_out, label="d(t)")

plt.xlabel("t")
plt.ylabel("E(t), d(t)")
plt.legend()

plt.savefig("dipole_t_abs.png")


# Define function to update plot for each frame of the animation
def update_plot(frame):
    plt.cla()
    plt.xlim([-20, 20])
    plt.ylim([0.0, 0.12])
    plt.plot(xj, np.abs(wavefunctions[frame])**2, label="Density")
    plt.xlabel('$x$')
    plt.ylabel('$Density$')
    plt.legend()

# Create the animation
fig = plt.figure()
ani = animation.FuncAnimation(fig, update_plot, frames=len(wavefunctions), interval=50)
#ani.save('wavefunction_animation.gif', writer='imagemagick')
ani.save('density_animation_abs.gif', writer='pillow')



