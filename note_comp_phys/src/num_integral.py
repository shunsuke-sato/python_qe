import numpy as np

a = 1.0
b = 2.0
n = 64

h = (b-a)/n

s = (1.0/a+1.0/b)/2.0
for i in range(1,n):
    x = a + i*h
    s += 1/x

s = s*h

print('num. integral = ',s)
print('log(2)        = ',np.log(2))
