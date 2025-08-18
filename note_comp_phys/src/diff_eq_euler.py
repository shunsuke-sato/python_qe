from matplotlib import pyplot as plt
import numpy as np

# compute dy/dx
def dydx(x,y):
    return -y

# compute y(x+h)
def euler_method(y,x,h):
    return y+dydx(x,y)*h


xi = 0.0
xf = 5.0
n = 10

h = (xf-xi)/n

# initial condition
x = xi
y = 1.0


x_j = np.zeros((n+1))
y_euler = np.zeros((n+1))
x_j[0] = x
y_euler[0] = y

for i in range(n):
    x = xi + i*h
    y = euler_method(y_euler[i],x,h)
    x_j[i+1] = x+h
    y_euler[i+1] = y


# plot
plt.plot(x_j, y_euler, label="Euler method")
plt.plot(x_j, np.exp(-x_j), label="Exact", linestyle='dashed')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig("result_Euler.png")
plt.show()

