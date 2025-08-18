import numpy as np


x = 0.0
h = 0.1

fx = np.exp(x)
fxph = np.exp(x+h)

num_dfdx = (fxph-fx)/h
ana_dfdx = np.exp(x)

error = np.abs(num_dfdx - ana_dfdx)

# Print results
print(f'For h = {h}:')
print(f'Numerical derivative of exp({x}) = {num_dfdx}')
print(f'Analytical derivative of exp({x}) = {np.exp(x)}')
print(f'Error = {error}')
