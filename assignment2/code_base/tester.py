import numpy as np

a = np.array([[1, 2, 3]])
b = np.array([[1], [2], [3]])
print a+b

c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print c.reshape(2, -1)

d = np.array([1, 2])
index = np.array([[1, 0], [0, 1]])
print d[index]
