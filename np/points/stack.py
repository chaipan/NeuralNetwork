import TensorFlow as tf
import numpy as np


a = range(27)
b = np.reshape(a, [3,3,3])


# print(a)
# print(b)
print(np.stack(b, axis=0))
print(np.stack(b, axis=1))
print(np.stack(b, axis=2))


# c = []
# c.append(np.random.rand(2,3))
# c.append(np.random.rand(2,3))
# c.append(np.random.rand(2,3))
# print(c)
# d = np.stack(c)
# print(d)
# print(np.stack(d, axis=2))
# print(np.stack(d, axis=1))
