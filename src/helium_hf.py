import numpy as np
from matplotlib import pyplot as plt


def set_grid(num_grid, radius):
    # define grid
    dr = radius/(num_grid+1)
    rj = np.linspace(dr, radius-dr, num_grid)

    return dr, rj



# Compute the density, rho, from the radial wavefunction, chi
def calc_density(chi, num_grid, radius):

    # define grid
    dr, rj = set_grid(num_grid, radius)
    
    # normalize chi
    norm = 4*np.pi*np.sum(chi**2)*dr
    chi[:] = chi[:]/np.sqrt(norm)
    
    # compute the full wavefunction
    phi = chi/rj

    # compute the electron density
    rho = phi**2

    return rho

# Compute the mean-field potential
def calc_potential(rho, num_grid, radius):
    
    # define grid
    dr, rj = set_grid(num_grid, radius)

    vmf_pot = np.zeros(num_grid)

    # compute the mean-field potential
    for ir in range(num_grid):
        
        v=0.0
        for ir2 in range(ir):
            v = v + rj[ir2]**2*rho[ir2]*dr

        v = v + 0.5*rj[ir]**2*rho[ir]*dr
        vmf_pot[ir] = 4.0*np.pi*v/rj[ir]

        v= 0.5*rj[ir]*rho[ir]*dr
        for ir2 in range(ir+1, num_grid):
            v = v + rj[ir2]*rho[ir2]*dr
            
        vmf_pot[ir] = vmf_pot[ir] + 4.0*np.pi*v

    return vmf_pot

def calc_hamiltonian(vmf_pot, num_grid, radius):

    # define grid
    dr, rj = set_grid(num_grid, radius)

    vpot = -2.0/rj + vmf_pot

    ham_mat = np.zeros((num_grid, num_grid))

    for i in range(num_grid):
        for j in range(num_grid):
            if(i == j):
                ham_mat[i,j] = -0.5*(-2.0/dr**2) + vpot[i]
            elif(np.abs(i-j) == 1):
                ham_mat[i,j] = -0.5*(1.0/dr**2)

    return ham_mat

def calc_energy(epsilon_gs, rho, vmf_pot, num_grid, radius):
    # define grid
    dr, rj = set_grid(num_grid, radius)

    energy = 2.0*epsilon_gs - 4.0*np.pi*np.sum(rj**2*rho*vmf_pot)*dr

    return energy


# number of scf loop
nscf = 20

# define grid
num_grid = 2048
radius = 20.0

# define grid
dr, rj = set_grid(num_grid, radius)

# initial guess
rho = np.zeros(num_grid)


for iscf in range(nscf):

    vmf_pot = calc_potential(rho, num_grid, radius)
    ham_mat = calc_hamiltonian(vmf_pot, num_grid, radius)
    eigenvalues, eigenvectors = np.linalg.eigh(ham_mat)
    epsilon_gs = eigenvalues[0]
    chi = eigenvectors[:,0]
    rho = calc_density(chi, num_grid, radius)

    energy = calc_energy(epsilon_gs, rho, vmf_pot, num_grid, radius)
    print("iscf, energy", iscf, energy)


print("Final energy =", energy, "Hartree", energy*27.2114, "eV")


