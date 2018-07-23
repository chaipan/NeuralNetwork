
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 50
LR = 0.001              # learning rate

mnist = input_data.read_data_sets('./mnist', one_hot=True)  # they has been normalized to range (0,1)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

# plot one example
print(mnist.train.images.shape)     # (55000, 28 * 28)
print(mnist.train.labels.shape)   # (55000, 10)
print(mnist.train.labels[0].shape)
for i in range(10):
	plt.imshow(mnist.train.images[i].reshape((28, 28)), cmap='gray')
	plt.title('%i' % np.argmax(mnist.train.labels[i]));
	plt.show()
	plt.pause(0.1)





# tf_x = tf.placeholder(tf.float32, [None, 28*28]) / 255.
# image = tf.reshape(tf_x, [-1, 28, 28, 1])              # (batch, height, width, channel)
# tf_y = tf.placeholder(tf.int32, [None, 10])            # input y
