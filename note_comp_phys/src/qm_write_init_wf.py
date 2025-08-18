from matplotlib import pyplot as plt
import numpy as np

# Initialize the wavefunction
def initialize_wf(xj, x0, k0, sigma0):
    wf = np.exp(1j*k0*(xj-x0))*np.exp(-0.5*(xj-x0)**2/sigma0**2)
    return wf


    
# initial wavefunction parameters
x0 = -25.0
k0 = 0.85
sigma0 = 5.0

# set the coordinate
xmin = -100.0
xmax = 100.0
n = 2500

dx = (xmax-xmin)/(n+1)
xj = np.zeros(n)

for i in range(n):
    xj[i] = xmin + dx*(i+1)


# Initialize the wavefunction
wf = initialize_wf(xj, x0, k0, sigma0)

# Plot the results
plt.plot(xj, np.real(wf), label="Real part")
plt.plot(xj, np.imag(wf), label="Imaginary part")

plt.xlabel('x')
plt.ylabel('$\psi$(x)')
plt.legend()
plt.savefig("init_wf.png")
plt.show()


    






