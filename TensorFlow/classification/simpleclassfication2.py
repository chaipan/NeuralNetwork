import tensorflow as tf
import numpy as np

# 造数据
x_feed = np.zeros([100, 10])
for i in range(x_feed.shape[0]):
    rand_index = np.random.randint(0, 10, 1)
    x_feed[i, rand_index] = 1
y_feed = np.argmax(x_feed, 1)

# 构造计算图
x = tf.placeholder(tf.float32, x_feed.shape)
y = tf.placeholder(tf.int32, y_feed.shape)
l1 = tf.layers.dense(x, 10, tf.nn.relu)
l2 = tf.layers.dense(l1, 10, tf.nn.relu)
out_put = tf.layers.dense(l2, 10)

loss = tf.losses.sparse_softmax_cross_entropy(y, out_put)
accuracy = tf.metrics.accuracy(tf.squeeze(y), tf.argmax(out_put, 1))[1]
optimizer = tf.train.AdamOptimizer(0.03)
train_op = optimizer.minimize(loss)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

sess = tf.Session()
sess.run(init_op)
for i in range(3000):
    _, acc = sess.run([train_op, accuracy], feed_dict={x:x_feed, y:y_feed})
    print(acc)

