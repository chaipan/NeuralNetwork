import TensorFlow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

# fake data造数据
# -1到1之间均匀生成100个数，排成一列
x = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.1, x.shape)
y = np.power(x, 2) + noise

# plot data
plt.scatter(x, y)  # 散点图
plt.show()


# 构建运算图
tf_x = tf.placeholder(tf.float32, x.shape)
tf_y = tf.placeholder(tf.float32, y.shape)
l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)
out_put = tf.layers.dense(l1, 1)

loss = tf.losses.mean_squared_error(tf_y, out_put)
train_op = tf.train.AdamOptimizer(0.1).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 打开交互模式
plt.ion()
for step in range(1000):
    _, l, pred = sess.run([train_op, loss, out_put], feed_dict={tf_x:x, tf_y:y})
    if step % 10 == 0:
        plt.cla()
        plt.scatter(x, y)
        plt.plot(x, pred, 'r_', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)


plt.ioff()
plt.show()


print(x)


