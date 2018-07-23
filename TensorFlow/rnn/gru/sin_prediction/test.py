import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt


a = np.arange(0, 100) / 10
b = np.sin(a + np.random.rand())

plt.scatter(a, b)
plt.show()
