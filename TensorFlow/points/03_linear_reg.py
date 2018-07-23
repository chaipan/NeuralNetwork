import os
import tensorflow as tf
import time
import numpy as np
import matplotlib.pylab as plt
from tensorflow.contrib import data as tfdata
from TensorFlow import utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_FILE = 'data/birth_life_2010.txt'

# step1: read in the data
data, n_samples = utils.read_birth_life_data(DATA_FILE)

# step2: create dataset and iterator
dataset = tfdata.Dataset.from_tensor_slices((data[:, 0], data[:, 1]))
iterator = dataset.make_initializable_iterator()
x, y = iterator.get_next()

# step3: cerate weight and biases and initialized to 0
w = tf.get_variable(name='weight', initializer=tf.constant(0.0))
b = tf.get_variable(name='biases', initializer=tf.constant(0.0))

# step4: create model to predisct Y
Y_predict = x * w + b

# step5: define the loss function (squared error)
loss = tf.square(y - Y_predict, name='loss')

# step6: optimize model,using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

start = time.time()
with tf.Session() as sess:
    # step7: initialize the variable
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs/lin_reg', sess.graph)

    # step 8: train the model for 100 epoched:
    for i in range(100):
        sess.run(iterator.initializer)
        total_loss = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
        except tf.errors.OutOfRangeError:
            pass
        print('Epochs:{0},{1}'.format(i, total_loss / n_samples))

    # close writer
    writer.close()

    # step9: output the value of w and b
    w_output, b_output = sess.run([w, b])
    print('w:{0:.3f}, b:{1:.3f}'.format(w_output, b_output))

print('took:{0:.3f} seconds '.format(time.time() - start))

# plot the results
plt.plot(data[:, 0], data[:, 1], 'bo', label='Real data')
plt.plot(data[:, 0], data[:, 0] * w_output + b_output, 'r', label='Predicted data with squared error')
# plt.plot(data[:,0], data[:,0] * (-5.883589) + 85.124306, 'g', label='Predicted data with Huber loss')
plt.legend()
plt.show()
