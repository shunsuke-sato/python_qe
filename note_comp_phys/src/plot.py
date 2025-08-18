from matplotlib import pyplot
import numpy as np


x = np.linspace(0.0, 10.0, 32)
y = np.cos(x)

pyplot.plot(x, y)
pyplot.savefig("plot.png")
pyplot.show()
