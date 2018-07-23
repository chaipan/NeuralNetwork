import TensorFlow as tf
import numpy as np

import TensorFlow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("D:/data/minist/", one_hot=True)

print("done")
# print("train dataset shapes {0}".format(mnist.train.images.shape))
# print("train labels shapes {0}".format(mnist.train.labels.shape))
# print("validate dataset shapes {0}".format(mnist.validation.images.shape))
# print("test dataset shapes {0}".format(mnist.test.images.shape))
#
# print("trainning data data element {0}".format((np.round(mnist.train.images[0].reshape(28, 28))), 1))

# 使用随机随机batch
batch_size = 100
X_train, Y_train = mnist.train.next_batch(batch_size)
print(X_train)

