import numpy as np
from matplotlib import pyplot as plt

# density parameters
rho0 = 1.0
a = 1.0

def set_grid(num_grid, radius):
    dr = radius/(num_grid + 1)
    # rj = dr*(j-1)
    # rj = 0 at j = -1
    # rj = radius at j = num_grid
    rj = np.linspace(dr, radius -dr , num_grid)

    return dr, rj

def set_rho(rj):
    rho = np.heaviside(a-rj, rho0)

    return rho


def calc_coulomb_pot(rj, rho):

    num_grid = rho.size
    dr = rj[1]-rj[0]
    vpot = np.zeros(num_grid)



    for i in range(num_grid):
        v1 = 0.0
        for j in range(i):
            v1 = v1 + rj[j]**2*rho[j]*dr

        v1 = v1 + 0.5*rj[i]**2*rho[i]*dr
        v1 = 4.0*np.pi*v1/rj[i]

        v2 = 0.5*rj[i]*rho[i]*dr
        for j in range(i+1, num_grid):
            v2 = v2 + rj[j]*rho[j]*dr
        
        v2 = 4.0*np.pi*v2

        vpot[i] = v1 + v2

    return vpot

def calc_coulomb_pot_exact(rj):
    num_grid = rj.size
    vpot_exact = np.zeros(num_grid)

    for i in range(num_grid):
        if(rj[i] < a):
            vpot_exact[i] = 4*np.pi*rho0*(3*a**2-rj[i]**2)/6
        else:
            vpot_exact[i] = (4/3)*np.pi*rho0*a**3/rj[i]

    return vpot_exact

# set parameters
radius = 10.0
num_grid = 100

dr, rj = set_grid(num_grid, radius)

rho = set_rho(rj)

vpot = calc_coulomb_pot(rj, rho)
vpot_exact = calc_coulomb_pot_exact(rj)


# plotting
plt.plot(rj, rho, label="rho(r)")
plt.plot(rj, vpot, label="vpot(r)")
plt.plot(rj, vpot_exact, label="vpot_exact(r)", linestyle="dashed")
plt.xlabel('r')
plt.ylabel('rho, vpot')
plt.legend()
plt.savefig('fig_vpot_coulomb.jpeg')
plt.show()



