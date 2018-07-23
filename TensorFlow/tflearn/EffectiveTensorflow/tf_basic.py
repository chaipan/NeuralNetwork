import TensorFlow as tf
import numpy as np



x = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)

w = tf.get_variable(name="w", shape=[2,1], dtype=tf.float32)

f_x = tf.stack([ x, tf.ones_like(x)], axis=1)
yhat = tf.matmul(f_x, w)

loss = tf.nn.l2_loss(y-yhat) + 0.1 * tf.nn.l2_loss(w)

train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

def generate_data():
    x_vals = np.random.uniform(size=100, low=-10.0, high=10.0)
    y_vals = x_vals + 5
    return x_vals, y_vals

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        x_val, y_val = generate_data()
        _, loss_val = sess.run([train_op, loss], {x:x_val, y:y_val})
        print(loss_val)
        # print(sess.run([w]))
    print(sess.run(w))