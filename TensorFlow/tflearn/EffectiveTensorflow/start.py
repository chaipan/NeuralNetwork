import numpy as np
import TensorFlow as tf


# normal为取正态分布样本函数，参数loc为期望，
x = np.random.normal(size=[10,10])
y = np.random.normal(size=[10,10])
z = np.dot(x, y)
print(z)

x = tf.random_normal([10, 10])
y = tf.random_normal([10,10])
z = tf.matmul(x, y)
print(z)