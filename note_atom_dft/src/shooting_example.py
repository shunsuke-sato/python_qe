import numpy as np
import matplotlib.pyplot as plt

# Define grid
radius = 20.0
num_grid = 2000
dr = radius / num_grid

rj = np.linspace(0.0, radius, num_grid)

# Define energy values
energy = np.array([-0.497, -0.499, -0.501, -0.503])
num_energy = energy.size

# Initialize chi array
chi = np.zeros((num_grid, num_energy))
chi[0, :] = 0.0
chi[1, :] = dr

# Compute wavefunction using finite difference method
factor = 2 * dr**2
for j in range(1, num_grid - 1):
    chi[j + 1, :] = (
        2 * chi[j, :]
        - chi[j - 1, :]
        - factor * (energy[:] + 1.0 / rj[j]) * chi[j, :]
    )

# Plot results
plt.figure(figsize=(8, 6))
for i, E in enumerate(energy):
    plt.plot(rj, chi[:, i], label=f"E={E:.3f} a.u.")

plt.xlim(0.0, 10.0)
plt.ylim(-1.5, 1.5)
plt.xlabel("Radius (Bohr)")
plt.ylabel(r"$\chi (r)$")
plt.title("Shooting Method Example")
plt.legend()
plt.grid()
plt.tight_layout()

# Save and show the figure
plt.savefig("shooting_example.pdf", dpi=300)
plt.show()
