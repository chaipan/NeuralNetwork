import tensorflow as tf
import numpy as np

data = np.random.rand(1,5,10)

print(data)



a = tf.placeholder(name="a", shape=[1, 5, 10], dtype=tf.float32)
x = tf.reshape(tensor=a, shape=[1, 50])

sess = tf.Session()
sess.run(x, {a:data})


# print(np.hstack((a, b)))
# sess = tf.Session()
# accuracy =  tf.metrics.accuracy(a,b)
# sess.run(tf.local_variables_initializer())
# print(sess.run(accuracy))
