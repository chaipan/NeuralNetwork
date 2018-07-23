import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data as input_data
from TensorFlow.cnn.LeNet5.lenet_new import inference as inference
# from TensorFlow.cnn.LeNet5.lenet_new import validate as validate
import matplotlib.pylab as plt



BATCH = 100
IMAGE_SIZE = 28
NUM_CHANNEL = 1
NUM_CLASSES = 10
EPOCH = 1000
LEARNING_RATE = 0.01


def train():
	data = input_data.read_data_sets("D://PyCharm//NeuralNetwork//TensorFlow//data//mnist", one_hot=True)


	# 构建计算图
	x = tf.placeholder(dtype=tf.float32, shape=[BATCH, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL], name="x")
	y = tf.placeholder(dtype=tf.float32, shape=[BATCH, NUM_CLASSES], name="y")
	logits = inference.inference(x)
	accuracy = tf.metrics.accuracy(tf.argmax(y,1), tf.argmax(logits,1))[1]
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y,1), logits=logits)
	loss = tf.reduce_mean(cross_entropy)

	validate_accuracy = validate(data)

	train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())



	sess = tf.Session()
	sess.run(init_op)
	accuracy_list = []
	plt.ion()

	for i in range(EPOCH):
		x_placeholder, y_placeholder = data.train.next_batch(BATCH)
		x_placeholder = np.reshape(x_placeholder, [BATCH, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
		y_placeholder = np.reshape(y_placeholder,[BATCH, NUM_CLASSES])
		_, loss_val, accuracy_val = sess.run([train_op, loss, accuracy], {x:x_placeholder, y:y_placeholder})
		if i % 10 == 0:
			accuracy_list.append(accuracy_val)
			# view(accuracy_list)
			print("step{0}, accuracy:{1:.5f}".format(i, accuracy_val))

		if i % 100 == 0:
			validate_accuracy_val = sess.run(validate_accuracy)
			print("step{0}, 验证集准确率为{1:.5f}".format(i, validate_accuracy_val))


	plt.ioff()
	view(accuracy_list)
	plt.show()



def view(accuracy_list):
	plt.plot(accuracy_list, 'r--' )
	plt.title("primitive version")
	plt.pause(0.1)


def validate(data):
	validation = data.validation
	accuracy_list = []
	for i in range(5):
		# np.random.shuffle(validation)
		x, y = validation.next_batch(1000)
		x_reshaped = np.reshape(x, [1000, 28, 28, 1])
		logits = inference.inference(x_reshaped)
		accucary = np.mean(np.equal(np.argmax(y, 1), np.argmax(logits, 1)))
		accuracy_list.append(accucary)
		print("数据验证， step{0}， 准确率{1:.5f}".format(i, accucary))


	return np.mean(accuracy_list)


if __name__ == "__main__":
	train()