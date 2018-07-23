


import TensorFlow as tf

import  numpy as np
import matplotlib.pyplot as plt


# 造数
x_tf = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.1, x_tf.shape)
y_tf = np.square(x_tf) + noise


# 构造图
x = tf.placeholder(tf.float32, x_tf.shape)
y = tf.placeholder(tf.float32, y_tf.shape)
l1 = tf.layers.dense(x, 10, tf.nn.relu)
yhat = tf.layers.dense(l1, 1)
loss = tf.losses.mean_squared_error(y, yhat)
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)





sess = tf.Session()
sess.run(tf.global_variables_initializer())
plt.ion()
for i in range(1000):
    _,loss_val,predict = sess.run([train_op, loss,yhat], feed_dict={x:x_tf, y: y_tf})
    plt.cla()
    plt.scatter(x_tf, y_tf)
    plt.plot(x_tf, predict)
    # plt.text()
    plt.show()
    plt.pause(0.1)




