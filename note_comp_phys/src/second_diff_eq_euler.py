from matplotlib import pyplot as plt
import numpy as np

def dxdt(v):
    return v

def dvdt(x):
    return -x

def euler_method(x,v,dt):
    x_updated = x + dt*dxdt(v)
    v_updated = v + dt*dvdt(x)
    return x_updated, v_updated

ti = 0.0
tf = 15.0
n = 45
dt = (tf - ti)/n

# Initial conditions
x = 1.0
v = 0.0


tt = np.zeros(n+1)
xt = np.zeros(n+1)
vt = np.zeros(n+1)

tt[0] = ti
xt[0] = x
vt[0] = v

for j in range(n):
    x, v = euler_method(x,v,dt)
    xt[j+1] = x
    vt[j+1] = v
    tt[j+1] = ti + (j+1)*dt


# Plot the results
plt.plot(tt, xt, label='x(t)')
plt.plot(tt, vt, label='v(t)')
plt.plot(tt, np.cos(tt), label='x(t): Exact', linestyle='dashed')
plt.plot(tt, -np.sin(tt), label='v(t): Exact', linestyle='dashed')

plt.xlabel('t')
plt.ylabel('x,v')
plt.legend()
plt.savefig("result_Euler_2nd.png")
plt.show()
