from matplotlib import pyplot as plt
import numpy as np

# Initialize the wavefunction
def initialize_wf(xj, x0, k0, sigma0):
    wf = np.exp(1j*k0*(xj-x0))*np.exp(-0.5*(xj-x0)**2/sigma0**2)
    return wf


# Operate the Hamiltonian to the wavefunction
def ham_wf(wf, vpot, dx):

    n = wf.size
    hwf = np.zeros(n, dtype=complex)
    
    for i in range(1,n-1):
        hwf[i] = -0.5*(wf[i+1]-2.0*wf[i]+wf[i-1])/(dx**2)

    i = 0
    hwf[i] = -0.5*(wf[i+1]-2.0*wf[i])/(dx**2)
    i = n-1
    hwf[i] = -0.5*(-2.0*wf[i]+wf[i-1])/(dx**2)
    
    hwf = hwf + vpot*wf
    
    return hwf

    
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
vpot = np.zeros(n)

hwf = ham_wf(wf, vpot, dx)

# Plot the results
plt.plot(xj, np.real(wf), label="Real part (wf)")
plt.plot(xj, np.imag(wf), label="Imaginary part (wf)")
plt.plot(xj, np.real(hwf), label="Real part (ham wf)")
plt.plot(xj, np.imag(hwf), label="Imaginary part (ham wf)")

plt.xlabel('x')
plt.ylabel('$\psi$(x)')
plt.legend()
plt.savefig("ham_wf.png")
plt.show()







