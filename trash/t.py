import numpy as np

a = np.arange(1000).reshape((10,10,10))
b = np.arange(1000).reshape((10,10,10))
c = np.arange(10)

np.random.shuffle(c)
a = a[c]
b = b[c]