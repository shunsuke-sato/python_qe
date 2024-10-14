import numpy as np
from matplotlib import pyplot as plt

hbar = 1.0
mass = 1.0
omega = 1.0

num_points = 128
alpha_min = 0.2
alpha_max = 5.0


alpha = np.linspace(alpha_min, alpha_max, num_points)
Ekin = alpha*hbar**2/(4.0*mass)
Vpot = mass*omega**2/(4.0*alpha)
Etot = Ekin + Vpot

plt.plot(alpha, Etot, label="Total energy")
plt.plot(alpha, Ekin, label="Kinetic energy")
plt.plot(alpha, Vpot, label="Potential energy")

plt.xlabel("Alpha")
plt.ylabel("Energy")

plt.legend()
plt.savefig('variational_harmonic_oscillator.jpeg')
plt.show()
