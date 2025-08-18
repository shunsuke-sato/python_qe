from matplotlib import pyplot as plt
import numpy as np

# Parameters
x = 0.0  # Point at which derivative is evaluated
step_sizes = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])  # Differentiation step sizes

# Numerical derivatives
forward_diff = (np.exp(x + step_sizes) - np.exp(x)) / step_sizes
central_diff = (np.exp(x + step_sizes) - np.exp(x - step_sizes)) / (2.0 * step_sizes)

# Error evaluation
error_forward = np.abs(forward_diff - np.exp(x))
error_central = np.abs(central_diff - np.exp(x))

# Plotting the data
plt.plot(step_sizes, error_forward, label="Forward Difference", marker='o')
plt.plot(step_sizes, error_central, label="Central Difference", marker='x')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Step Size (h)')
plt.ylabel('Error')
plt.title('Error in Numerical Differentiation')
plt.legend()
plt.grid(True)

# Saving the plot
plt.savefig("error_derivative.png")
plt.show()
