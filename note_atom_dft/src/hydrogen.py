import numpy as np
from matplotlib import pyplot as plt

def calc_radial_wavefunction(zval, l, dr, rmax, num_state):
    num_grid = int(rmax / dr) + 1
    rj = np.linspace(0.0, rmax, num_grid)

    chi = np.zeros((num_grid, num_state))
    energy = np.zeros(num_state)

    for jstate in range(num_state):
        chi[:, jstate], energy[jstate] = shooting_method(zval, l, dr, num_grid, rj, jstate)

    return rj, chi, energy


def shooting_method(zval, l, dr, num_grid, rj, jstate):
    chi_s = np.zeros(num_grid)
    ene_max = 0.1 * zval**2
    ene_min = -0.6 * zval**2

    for iter in range(100):
        ene_t = 0.5 * (ene_max + ene_min)

        chi_s, num_node = get_radial_wavefunction(zval, l, dr, num_grid, rj, jstate, ene_t)

        if num_node >= jstate+1:
            ene_max = ene_t
        else:
            ene_min = ene_t

        if ene_max - ene_min < 1e-6:
            break


    ene_t = ene_max
    chi_s, num_node = get_radial_wavefunction(zval, l, dr, num_grid, rj, jstate, ene_t)

# refine wavefunction
    num_node = 0
    for j in range(1, num_grid - 1):
        if chi_s[j + 1] == 0.0:
            num_node += 1
        elif chi_s[j + 1] * chi_s[j] < 0.0:
            num_node += 1

        if num_node == jstate+1:
            chi_s[j+1:] = 0.0
            break

    norm = np.sum(chi_s**2)*dr
    chi_s = chi_s / np.sqrt(norm)

    return chi_s, ene_t


def get_radial_wavefunction(zval, l, dr, num_grid, rj, jstate, energy):
    chi_s = np.zeros(num_grid)
    chi_s[0] = 0.0
    chi_s[1] = dr / zval
    factor = 2 * dr**2

    num_node = 0

    for j in range(1, num_grid - 1):
        chi_s[j + 1] = (
            2 * chi_s[j] - chi_s[j - 1]
            - factor * (energy - 0.5 * l * (l + 1) / rj[j]**2 + zval / rj[j]) * chi_s[j]
        )

        if chi_s[j+1] == 0.0:
            num_node += 1
        elif chi_s[j+1] * chi_s[j] < 0.0:
            num_node += 1

    return chi_s, num_node

zval = 1.0
num_state = 3
dr = 0.01
rmax = 100.0
l_angular = np.array([0, 1, 2])

for l in l_angular:
    rj, chi_l, energy = calc_radial_wavefunction(zval, l, dr, rmax, num_state)
    print(f"l = {l}, Energy levels : {energy}")

    # Plot results
    plt.figure(figsize=(8, 6))
    for i in range(num_state):
        plt.plot(rj, chi_l[:, i], label=f"{i}-state")

    plt.xlim(0.0, 50.0)
    plt.ylim(-0.5, 0.8)
    plt.xlabel("Radius (Bohr)")
    plt.ylabel(r"$\chi (r)$")
    plt.title(f"Wavefunctions (l={l})")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(f"hydrogen_wf_l{l}.pdf", dpi=300)

