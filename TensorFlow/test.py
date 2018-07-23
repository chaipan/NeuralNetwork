import numpy as np
import TensorFlow as tf
import matplotlib.pylab as plt


samples = np.random.randint(low=0,high=2, size=[10000,2])

def a(x_input):
    w = tf.get_variable(name="w", shape=[2,10], dtype=tf.float32, initializer=tf.truncated_normal_initializer)
    b = tf.get_variable(name="b", shape=10, dtype=tf.float32, initializer=tf.constant_initializer)
    w2 = tf.get_variable(name="w2", shape=[10,10], dtype=tf.float32, initializer=tf.truncated_normal_initializer)
    b2 = tf.get_variable(name="b2", shape=10, dtype=tf.float32, initializer=tf.constant_initializer)
    w3 = tf.get_variable(name="w3", shape=[10,1], dtype=tf.float32, initializer=tf.truncated_normal_initializer)
    b3 = tf.get_variable(name="b3", shape=1, dtype=tf.float32, initializer=tf.constant_initializer)

    layer1 = tf.nn.sigmoid(tf.matmul(x_input, w) + b)
    layer2 = tf.nn.sigmoid(tf.matmul(layer1, w2) + b2)
    layer3 = tf.matmul(layer2, w3) + b3

    return layer3



if __name__ == "__main__":
    x = tf.placeholder(shape=[10,2], dtype=tf.float32)
    layer3 = a(x)
    loss = tf.squared_difference(tf.reshape(tf.reduce_max(x, axis=1), shape=(10,1)), layer3)
    loss = tf.reduce_mean(loss)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        start = 0
        end = start + 10
        loss_array = []
        for i in range(400):
            y, y_ = sess.run([tf.reduce_max(x,axis=1), tf.reshape(layer3, shape=(1,10))], feed_dict={x: samples[start:end,:]})
            _, cost = sess.run([train_step, loss], feed_dict={x: samples[start:end,:]})
            loss_array.append(cost)
            start = end
            end = start + 10
            print(cost)
            print("y",y,"y_", y_)

        plt.plot(np.arange(400), loss_array)
        plt.show()