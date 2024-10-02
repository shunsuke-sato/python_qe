import numpy as np

def potential(r):
    v_pot = -1.0/r
    #v_pot = (1.0/2.0)*r**2
    #v_pot = (1.0/2.0)*r**4
    return v_pot

def calc_expectation_values(u, num_grid, radius):
    dr = radius/num_grid
    rj = np.linspace(dr, radius-dr, num_grid)
    v_pot = potential(rj)
    
# Normalize wavefunction
    norm = np.sum(u**2)*dr
    u = u/np.sqrt(norm)
    
    Tu = np.zeros(num_grid)
    
    for i in range(num_grid):
        if(i==0):
            Tu[i] = -0.5*(u[i+1]-2.0*u[i])/dr**2
        elif(i==num_grid-1):
            Tu[i] = -0.5*(-2.0*u[i]+u[i-1])/dr**2
        else:
            Tu[i] = -0.5*(u[i+1]-2.0*u[i]+u[i-1])/dr**2
    
            
    kinetic_energy = np.sum(u*Tu)*dr
    potential_energy = np.sum(v_pot*u**2)*dr
    
    return kinetic_energy, potential_energy

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
            ham_mat[i,j] = -0.5*(-2.0/dr**2) + potential(rj[j])
        elif(np.abs(i-j) == 1):
            ham_mat[i,j] = -0.5*(1.0/dr**2)


# Calculate eigenvectors and eigenvalues
eigenvalues , eigenvectors = np.linalg.eigh(ham_mat)

# Print the energy and decomposition
u = eigenvectors

# ground state
kinetic_energy, potential_energy = calc_expectation_values(u[:,0], num_grid, radius)
print('Energy (1s)      =', eigenvalues[0])
print('Kinetic energy  =', kinetic_energy)
print('Potential energy=', potential_energy)
print('Virial ratio    =', kinetic_energy/potential_energy)
print()


# 1st excited state
kinetic_energy, potential_energy = calc_expectation_values(u[:,1], num_grid, radius)
print('Energy (2s)      =', eigenvalues[1])
print('Kinetic energy  =', kinetic_energy)
print('Potential energy=', potential_energy)
print('Virial ratio    =', kinetic_energy/potential_energy)
print()
