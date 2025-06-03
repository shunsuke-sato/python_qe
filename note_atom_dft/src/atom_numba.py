from numba import jit
import numpy as np
from matplotlib import pyplot as plt

def calc_atom_gs(orb_name, orb_occ, nscf, rmax, dr, zval):

    num_orb, l_ang_mom, occupation, num_node  = list_of_orbitals(orb_name, orb_occ)
    num_grid = int(rmax/dr)+1
    rj = np.linspace(0.0, rmax, num_grid)
    phi = np.zeros((num_grid, num_orb))
    rho_old = np.zeros(num_grid)
    update_rate = 0.5
    
    for iscf in range(nscf):

        rho = calc_density(num_orb, phi, occupation)
        rho = update_rate*rho + (1.0 - update_rate)*rho_old
        rho_old = rho
        
        vhxc, vh, vxc = calc_potential(rho, rj, dr, num_grid)
        phi, epsilon_s = calc_wavefunction(rj, dr, num_grid, vhxc, zval, num_orb, l_ang_mom, num_node)

        total_energy = calc_total_energy(rj, dr, num_grid, phi, epsilon_s, rho, num_orb, vhxc, vh, occupation)

        print("iscf, energy =",iscf, total_energy, "Hartree")

    return num_orb, rj, phi, epsilon_s, total_energy

def calc_wavefunction(rj, dr, num_grid, vhxc, zval, num_orb, l_ang_mom, num_node):

    phi = np.zeros((num_grid, num_orb))
    epsilon_s = np.zeros(num_orb)
    
    for iorb in range(num_orb):

        phi_t, ene_t = shooting_method(rj, dr, num_grid, vhxc, zval, l_ang_mom[iorb], num_node[iorb])
        phi[:,iorb] = phi_t[:]
        epsilon_s[iorb] = ene_t

    return phi, epsilon_s


def list_of_orbitals(orb_name, orb_occ):
    num_orb = len(orb_name)

    l_ang_mom = np.zeros(num_orb)
    occupation  = np.zeros(num_orb)
    num_node = np.zeros(num_orb, dtype=int)

    num_node_s = 0
    num_node_p = 0
    num_node_d = 0
    num_node_f = 0

    for iorb in range(num_orb):
        occupation[iorb] = orb_occ[iorb]
        if('s' in orb_name[iorb].lower()):
            l_ang_mom[iorb] = 0.0
            num_node[iorb] = num_node_s
            num_node_s = num_node_s + 1
        elif('p' in orb_name[iorb].lower()):
            l_ang_mom[iorb] = 1.0
            num_node[iorb] = num_node_p
            num_node_p = num_node_p + 1
        elif('d' in orb_name[iorb].lower()):
            l_ang_mom[iorb] = 2.0
            num_node[iorb] = num_node_d
            num_node_d = num_node_d + 1
        elif('f' in orb_name[iorb].lower()):
            l_ang_mom[iorb] = 3.0
            num_node[iorb] = num_node_f
            num_node_f = num_node_f + 1

    return num_orb, l_ang_mom, occupation, num_node

@jit(nopython=True)    
def calc_density(num_orb, phi, occupation):

    rho = np.zeros(phi.shape[0])
    for iorb in range(num_orb):
        rho += occupation[iorb]*phi[:,iorb]**2

    return rho
@jit(nopython=True)
def calc_potential(rho, rj, dr, num_grid):

    vhxc = np.zeros(num_grid)
    vh = np.zeros(num_grid)
    
    for i in range(num_grid):

        v1 = 0.0
        for j in range(i):
            v1 = v1 + rj[j]**2*rho[j]*dr

        v1 = v1 + 0.5*rj[i]**2*rho[i]*dr
        if(i == 0):
            v1 = 0.0
        else:
            v1 = 4.0*np.pi*v1/rj[i]


        v2 = 0.5*rj[i]*rho[i]*dr
        for j in range(i+1, num_grid):
            v2 = v2 + rj[j]*rho[j]*dr

        v2 = 4.0*np.pi*v2

        vh[i] = v1 + v2


    vxc = -((3.0/np.pi)*rho)**(1.0/3.0)
    vhxc = vh + vxc

    return vhxc, vh, vxc



def shooting_method(rj, dr, num_grid, vhxc, zval, l_ang_mom_in, num_node_in):
    
    ene_max = 0.1 * zval**2
    ene_min = -0.6 * zval**2


    for iter in range(100):
        ene_t = 0.5 * (ene_max + ene_min)

        chi, num_node_t = get_wavefunction(rj, dr, num_grid, vhxc, zval, ene_t, l_ang_mom_in)

        if num_node_t >= num_node_in+1:
            ene_max = ene_t
        else:
            ene_min = ene_t

        if ene_max - ene_min < 1e-6:
            break

    ene_t = ene_max
    chi, num_node_t = get_wavefunction(rj, dr, num_grid, vhxc, zval, ene_t, l_ang_mom_in)

    # refine wavefunction
    num_node_t = 0
    for j in range(1, num_grid - 1):
        if chi[j + 1] == 0.0:
            num_node_t += 1
        elif chi[j + 1] * chi[j] < 0.0:
            num_node_t += 1

        if num_node_t == num_node_in + 1:
            chi[j+1:] = 0.0
            break

    norm = np.sum(chi**2)*dr
    chi = chi/np.sqrt(norm)

    phi_t = np.zeros(num_grid)
    phi_t[1:num_grid-1] = chi[1:num_grid-1]/rj[1:num_grid-1]
    phi_t[0] = 2*phi_t[1] - phi_t[2]
    phi_t = phi_t/(np.sqrt(4.0*np.pi))


    return phi_t, ene_t
@jit(nopython=True)        
def get_wavefunction(rj, dr, num_grid, vhxc, zval, energy, l_ang):

    chi = np.zeros(num_grid)
    num_node = 0


    chi[0] = 0.0
    chi[1] = dr / zval

    factor = 2 * dr**2
    for j in range(1, num_grid - 1):
        potential = - zval / rj[j] + vhxc[j] + 0.5*l_ang*(l_ang+1.0)/rj[j]**2
        chi[j + 1] = (
            2 * chi[j] - chi[j - 1]
            - factor * (energy - potential) * chi[j]
        )

        if chi[j+1] == 0.0:
            num_node += 1
        elif chi[j+1] * chi[j] < 0.0:
            num_node += 1

    return chi, num_node

@jit(nopython=True)
def calc_total_energy(rj, dr, num_grid, phi, epsilon_s, rho, num_orb, vhxc, vh, occupation):
    total_energy = np.sum(occupation*epsilon_s)
    total_energy = total_energy - 4.0*np.pi*np.sum(rho*vhxc*rj**2)*dr
    total_energy = total_energy + 0.5*4.0*np.pi*np.sum(rho*vh*rj**2)*dr
    total_energy = total_energy - 4.0*np.pi*(3.0/4.0)*(3.0/np.pi)**(1.0/3.0)*np.sum(rho**(4.0/3.0)*rj**2)*dr

    return total_energy
    

zval = 10.0
nscf = 30
rmax = 20.0
dr = 0.005

orb_name = ['1s', '2s', '2p' ]
orb_occ  = [2.0, 2.0, 6.0 ]


num_orb, rj, phi, epsilon_s, total_energy = calc_atom_gs(orb_name, orb_occ, nscf, rmax, dr, zval)

# The conversion factor from Hartree to electronvolt (eV)
ev = 27.2114

print("Total energy =", total_energy, "Hartree", total_energy*ev, "eV")
print("Single particle energies")
for iorb in range(num_orb):
    print(orb_name[iorb], ":", epsilon_s[iorb], "Hartree", epsilon_s[iorb]*ev, "eV")



