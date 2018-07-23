import tensorflow as tf
import numpy as np
import TensorFlow.cnn.LeNet5.inference as inference
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib as contrib

BATCH_SIZE = 100
LEARNING_RATE = 0.01
DECAY_RATE = 0.99
EMA_DECAY = 0.99
REGULARIZER_RATE = 0.0001


def train(mnist):

	# 数据为传入参数

	# 构造计算图]
	# LeNet的第一层就是卷积层，所以输入数据是[batch, size, size, channels]的形式
	x = tf.placeholder(shape=[BATCH_SIZE, inference.IMAGE_SIEZ, inference.IMAGE_SIEZ, inference.IMAGE_CHANNELS],
					   dtype=tf.float32)
	y = tf.placeholder(shape=[BATCH_SIZE],
					   dtype=tf.int32)
	l2_regularizer = contrib.layers.l2_regularizer(REGULARIZER_RATE)
	logits = inference.inference(x, l2_regularizer)
	accuracy = tf.metrics.accuracy(y, tf.argmax(logits, 1))

	# 损失函数定义
	cross_entropy = tf.losses.sparse_softmax_cross_entropy(y, logits)
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	# 对全连接层使用正则化
	loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

	# 对学习率使用指数衰减
	global_step = tf.Variable(0, trainable=False)
	learning_rate_decayd = tf.train.exponential_decay(LEARNING_RATE, global_step, BATCH_SIZE, DECAY_RATE)
	optimizer = tf.train.AdamOptimizer(learning_rate_decayd)

	# global_step在经过minimize后自动增长1
	train_op = optimizer.minimize(loss, global_step)

	# 对变量使用滑动平均模型
	ema_op = tf.train.ExponentialMovingAverage(EMA_DECAY, global_step).apply(tf.trainable_variables())
	with tf.control_dependencies([train_op, ema_op]):
		train = tf.no_op()

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	for step in range(100):
		images, labels = mnist.train.next_batch(BATCH_SIZE)
		images = np.reshape(images, [BATCH_SIZE, inference.IMAGE_SIEZ, inference.IMAGE_SIEZ, inference.IMAGE_CHANNELS])
		_, loss_val, accuracy_val, global_step_val = sess.run([train, loss, accuracy, global_step], {x:images, y:labels})
		print("step{0}, accuracy is {1}".format(global_step_val, accuracy_val[1]))


if __name__ == "__main__":
	mnist = input_data.read_data_sets("D://Documents//mnist")
	train(mnist)