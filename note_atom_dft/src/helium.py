import numpy as np
from matplotlib import pyplot as plt

def calc_hf_method(nscf, rmax, dr, zval):

    num_grid = int(rmax/dr)+1
    rj = np.linspace(0.0, rmax, num_grid)

    phi = np.zeros(num_grid)

    for iscf in range(nscf):

        rho = phi**2
        vhf = calc_potential(rho, rj, dr, num_grid)
        phi, epsilon_s = calc_wavefunction(rj, dr, num_grid, vhf, zval)

        total_energy = 2*epsilon_s - 4.0*np.pi*np.sum(rj**2*rho*vhf)*dr

        print("iscf, energy =",iscf, total_energy)

    return rj, phi, total_energy

def calc_potential(rho, rj, dr, num_grid):

    vhf = np.zeros(num_grid)
    
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

        vhf[i] = v1 + v2

    return vhf

def calc_wavefunction(rj, dr, num_grid, vhf, zval):
    
    ene_max = 0.0 * zval
    ene_min = -0.6 * zval**2


    for iter in range(100):
        ene_t = 0.5 * (ene_max + ene_min)

        chi, num_node = get_wavefunction(rj, dr, num_grid, vhf, zval, ene_t)

        if num_node >= 1:
            ene_max = ene_t
        else:
            ene_min = ene_t

        if ene_max - ene_min < 1e-6:
            break

    ene_t = ene_max
    chi, num_node = get_wavefunction(rj, dr, num_grid, vhf, zval, ene_t)

    # refine wavefunction
    num_node = 0
    for j in range(1, num_grid - 1):
        if chi[j + 1] == 0.0:
            num_node += 1
        elif chi[j + 1] * chi[j] < 0.0:
            num_node += 1

        if num_node == 1:
            chi[j+1:] = 0.0
            break

    norm = np.sum(chi**2)*dr
    chi = chi/np.sqrt(norm)

    phi = np.zeros(num_grid)
    phi[1:num_grid-1] = chi[1:num_grid-1]/rj[1:num_grid-1]
    phi[0] = 2*phi[1] - phi[2]
    phi = phi/(np.sqrt(4.0*np.pi))


    return phi, ene_t
        
def get_wavefunction(rj, dr, num_grid, vhf, zval, energy):

    chi = np.zeros(num_grid)
    num_node = 0

    chi[0] = 0.0
    chi[1] = dr / zval

    factor = 2 * dr**2
    for j in range(1, num_grid - 1):
        chi[j + 1] = (
            2 * chi[j] - chi[j - 1]
            - factor * (energy + zval / rj[j] - vhf[j]) * chi[j]
        )

        if chi[j+1] == 0.0:
            num_node += 1
        elif chi[j+1] * chi[j] < 0.0:
            num_node += 1

    return chi, num_node

zval = 2.0
nscf = 10
rmax = 20.0
dr = 0.005


rj, phi, total_energy = calc_hf_method(nscf, rmax, dr, zval)


chi = rj*phi
plt.figure(figsize=(8, 6))
plt.plot(rj, chi)
plt.xlim(0.0, 8.0)
plt.ylim(-0.1, 0.3)
plt.xlabel("Radius (Bohr)")
plt.ylabel(r"$\chi (r)$")
plt.title(f"Wavefunctions")
plt.grid()
plt.tight_layout()

plt.savefig("helium_hf_wavefunction.pdf", dpi=300)

