import tensorflow as tf
import numpy as np


global_step = 0
ema = tf.train.ExponentialMovingAverage(0.99, global_step)

a = tf.Variable(initial_value=0.0, trainable=True)

ema_op = ema.apply([a])

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for i in range(10000):
		rate = min(0.99, (global_step + 1.0) / (global_step + 10.0))
		sess.run(ema_op)
		print("step %d, a = %d, rate=%.4f, average=%.5f"%(global_step,sess.run(a), rate, sess.run(ema.average(a))))
		# print(sess.run(a) * (1-rate) + (1-rate)*)
		global_step += 1
		sess.run(tf.assign(a, a + 3))




# for i in range(1000):
# 	print(np.power(0.99, i))