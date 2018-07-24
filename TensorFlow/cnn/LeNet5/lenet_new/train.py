import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data as input_data
from TensorFlow.cnn.LeNet5.lenet_new import inference as inference
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

	# 学习率加入加入指数衰减
	global_step = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)
	decayed_learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE, global_step=global_step, decay_steps=10, decay_rate=0.99)
	train_op = tf.train.AdamOptimizer(decayed_learning_rate).minimize(loss)

	# 加入滑动平均模型
	ema = tf.train.ExponentialMovingAverage(decay=0.99, num_updates=global_step)
	ema_op = ema.apply(tf.trainable_variables())
	# 使用依赖控制将训练和滑动平均模型绑定在一起， 每训练一次执行一次
	with tf.control_dependencies([train_op, ema_op]):
		train_step = tf.no_op()

	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

	sess = tf.Session()
	sess.run(init_op)
	accuracy_list = []
	plt.ion()

	for i in range(EPOCH):
		x_placeholder, y_placeholder = data.train.next_batch(BATCH)
		x_placeholder = np.reshape(x_placeholder, [BATCH, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
		y_placeholder = np.reshape(y_placeholder,[BATCH, NUM_CLASSES])
		_, loss_val, accuracy_val = sess.run([train_step, loss, accuracy], {x:x_placeholder, y:y_placeholder})
		if i % 10 == 0:
			accuracy_list.append(accuracy_val)
			# view(accuracy_list)
			print("step{0}, accuracy:{1:.5f}".format(i, accuracy_val))




	plt.ioff()
	view(accuracy_list)
	plt.show()



def view(accuracy_list):
	plt.plot(accuracy_list, 'r--' )
	plt.title("primitive version")
	plt.pause(0.1)




















"""
在一次计算中只能使用一个计算模型，不能多次出现variable = inference.inference()
在placeholder中使用shape=[None, 数字]的组合时，若使用tf.reshape，不管是对该站位量使用还是他参与计算的后续量使用都会提示类型转换错误。
"""
# def validate(data):
# 	if i % 100 == 0:
# 		validation_accuracy_list = []
# 		for j in range(5):
# 			# np.random.shuffle(validation)
# 			validation_x, validation_y = data.validation.next_batch(1000)
# 			validation_x_reshaped = np.reshape(validation_x, [1000, 28, 28, 1])
# 			validation_logits_val = sess.run(validation_logits, {validation_x_placeholder: validation_x_reshaped})
# 			validation_accucary = np.mean(np.equal(np.argmax(validation_logits_val, 1), np.argmax(validation_y, 1)))
# 			validation_accuracy_list.append(validation_accucary)
# 			print("数据验证， step{0}， 准确率{1:.5f}".format(j, validation_accucary))
# 		validate_accuracy_mean = np.mean(validation_accuracy_list)
# 		print("step{0}, 验证集准确率为{1:.5f}".format(i, validate_accuracy_mean))
# 	pass



if __name__ == "__main__":
	train()