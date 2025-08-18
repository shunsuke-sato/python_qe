import numpy as np
from matplotlib import pyplot as plt

# Constants
dx = 0.1
nx = 128
x_start = -6.3
x_end = 6.3

# Generate x values
x = np.linspace(x_start, x_end, nx)

# Compute function values
fx = np.cos(x)
fx_pdx = np.cos(x + dx)
fx_mdx = np.cos(x - dx)

# Second derivative using central difference
d2fdx2 = (fx_pdx - 2 * fx + fx_mdx) / dx**2

# Plotting the function and its second derivative
plt.plot(x, fx, label="f(x) = cos(x)")
plt.plot(x, d2fdx2, label="f''(x)")

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function and Second Derivative')
plt.legend()
plt.savefig("second_derivative_cos.png")
plt.show()
