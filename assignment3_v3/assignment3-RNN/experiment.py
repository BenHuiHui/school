import numpy as np

a = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

b = np.array([1, 1, 1])

c = np.zeros((2, 3))
c[0, :] = b

print c
print a[0].shape

d = (0, 1)
print d[0]
