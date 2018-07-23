import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import TensorFlow as tf
import time
import numpy as np
import matplotlib.pylab as plt
import TensorFlow.utils as utils

DATA_FILE = 'data/birth_life_2010.txt'

# step1: read data in file data_file
data, n_samples = utils.read_birth_life_data(DATA_FILE)

# step2: create placeholders for x(birth rate) and y (life expactancys)
x = tf.placeholder(dtype=tf.float32, name='x')
y = tf.placeholder(dtype=tf.float32, name='y')

# step3: create weight and biases , initialised to 0
w = tf.get_variable(name='w', initializer=tf.constant(0.0), dtype=tf.float32)
b = tf.get_variable(name='b', initializer=tf.constant(0.0), dtype=tf.float32)

# step4: build model to predict y
y_predict = w * x + b

# step5: use the squared error ass the loss function
loss = tf.square(y - y_predict, name='loss')

# step6 :using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

start = time.time()
writer = tf.summary.FileWriter('./graphs/linear', tf.get_default_graph())
with tf.Session() as sess:
    # step7: initialize variable
    sess.run(tf.global_variables_initializer())

    # step8: train the model for 100 epoches:
    for i in range(100):
        total_loss = 0
        for X, Y in data:
            _, l = sess.run([optimizer, loss], feed_dict={x: X, y: Y})
        total_loss += l
        print('Epoches{0}:{1}'.format(i, total_loss/n_samples))

        # close the writer when you are done using it
        writer.close()

        # step9: output the value of w and b
        w_out, b_out = sess.run([w, b])

print('Took: {0:.3f} seconds '.format(time.time() - start))


# plot the results
plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data')
plt.legend()
plt.show()




