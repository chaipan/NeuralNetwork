import TensorFlow as tf
import numpy as np
import matplotlib.pyplot as plt

# 造数据
"""
构造n行两列的数据，每一行有一个分类结果，分类依据是两列数据根据所遵循的随机分布规律不同


"""
x_0 = np.random.normal(0, 0.1, [100,2])
y_0 = np.zeros(100)
x_1 = np.random.normal(2,0.1,[100,2])
y_1 = np.ones(100)
x_feed = np.vstack((x_0, x_1))
y_feed = np.hstack((y_0, y_1))
plt.scatter(x_feed[:, 0], x_feed[:,1])
plt.show()


x = tf.placeholder(tf.float32, x_feed.shape)
y = tf.placeholder(tf.int32, y_feed.shape)
l1 = tf.layers.dense(x, 10, tf.nn.relu)
l2 = tf.layers.dense(l1, 2)
out_put = l2
loss = tf.losses.sparse_softmax_cross_entropy(y, out_put)
optimizer = tf.train.GradientDescentOptimizer(0.05)
train_op = optimizer.minimize(loss)
# accuracy 函数
accurancy = tf.metrics.accuracy(tf.squeeze(y), tf.argmax(out_put, 1))[1]

init_op = tf.group((tf.global_variables_initializer(), tf.local_variables_initializer()))


sess = tf.Session()


sess.run(init_op)
for step in range(1000):
    _, loss_val, acc = sess.run([train_op, loss, accurancy], feed_dict={x:x_feed, y:y_feed})
    # print(sess.run(out_put, feed_dict={x:x_feed, y:y_feed}))
    print("accuracy is %.2f"%(acc))


