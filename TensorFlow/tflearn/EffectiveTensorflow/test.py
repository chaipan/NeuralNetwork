import TensorFlow as tf
import numpy as np


a = np.random.randint(0,2,size=(10,1,2))
b = np.random.randint(0,2,size=(10,1))
print(a)
print(np.squeeze(a))



# print(np.hstack((a, b)))
# sess = tf.Session()
# accuracy =  tf.metrics.accuracy(a,b)
# sess.run(tf.local_variables_initializer())
# print(sess.run(accuracy))
