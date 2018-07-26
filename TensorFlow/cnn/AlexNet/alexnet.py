import tensorflow as tf
import numpy as np


class alexNet(object):

	def __init__(self, x, keep_prop, classNum, skip, modelPath):
		self.X = x
		self.KEEP_PROP = keep_prop
		self.CLASSNUM = classNum
		self.SKIP = skip
		self.batch = x.get_shape()[0]
		self.input_channel = x.get_shape()[3]



	"""
		卷积层1， 11的核
	"""
	def buildCNN(self):
		with tf.variable_scope("conv1"):
			conv1_weight = tf.get_variable(name="conv1_weight",
										   shape=[11,11,self.input_channel, 96],
										   dtype=tf.float32,
										   initializer=tf.truncated_normal_initializer)
			conv1_biases = tf.get_variable(name="conv2_weight",
										   shape=[96],
										   dtype=tf.float32,
										   initializer=tf.constant_initializer(0.01))
			conv1 = tf.nn.conv2d(self.X, filter=conv1_weight,
								 strides=[1,4,4,1],
								 padding="SAME")
			# 现在一般认为LRN没有效果，所以不用LRN
			conv1_relu1 = tf.nn.relu(tf.nn.bias_add(conv1_weight, conv1_biases))
			conv1_pooling = tf.nn.max_pool(conv1_relu1,
										   ksize=[1,3,3,1],
										   strides=[1,2,2,1],
										   padding="SAME")

		with tf.variable_scope("conv2"):
			conv2_weight = tf.get_variable(name="conv2_weight",
										   shape=[5,5,96,256],
										   dtype=tf.float32,
										   initializer=tf.truncated_normal_initializer)
			conv2_biases = tf.get_variable(name="conv2_biases",
										   shape=[256],
										   dtype=tf.float32,
										   initializer=tf.constant_initializer(0.01))
			# 此处又一次局部归一化，略去
			conv2 = tf.nn.conv2d(conv1_pooling, filter=conv2_weight, strides=[1,1,1,1],padding="SAME")
			conv2_relu = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
			conv2_pooling = tf.nn.max_pool(value=conv2_relu, ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")

		with tf.get_variable("conv3"):
			conv3_weight = tf.get_variable(name="conv3_weight",
										   shape=[3,3,256,384],
										   dtype=tf.float32,
										   initializer=tf.truncated_normal_initializer)
			conv3_biases = tf.get_variable(name="conv3_biases",
										   shape=[384],
										   dtype=tf.float32,
										   initializer=tf.constant_initializer(0.01))
			conv3 = tf.nn.conv2d(conv2_relu, filter=conv3_weight, strides=[1,1,1,1], padding="SAME")
			conv3_relu = tf.nn.relu(tf.nn.bias_add((tf.nn.bias_add(conv3, conv3_biases))))

		with tf.variable_scope("conv4"):
			conv4_weight = tf.get_variable("conv4_weight",
										   shape=[3,3,384,384],
										   dtype=tf.float32,
										   initializer=tf.truncated_normal_initializer)
			conv4_biases = tf.get_variable("conv4_biases",
										   shape=[384],
										   dtype=tf.float32,
										   initializer=tf.constant_initializer(0.01))
			conv4 = tf.nn.conv2d(conv3_relu, filter=conv4_weight,strides=[1,1,1,1], padding="SAME")
			conv4_relu = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

		with tf.variable_scope("conv5"):
			conv5_weight = tf.get_variable("conv5_weight",
										   shape=[3,3,384,256],
										   dtype=tf.float32,
										   initializer=tf.truncated_normal_initializer)
			conv5_biases = tf.get_variable("conv5_biases",
										   shape=[256],
										   dtype=tf.float32,
										   initializer=tf.constant_initializer(0.01))
			conv5 = tf.nn.conv2d(conv4_relu,filter=conv5_weight,
								 strides=[1,1,1,1],
								 padding="SAME")
			conv5_relu = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases))
			conv5_pool = tf.nn.max_pool(value=conv5_relu,
										ksize=[1,3,3,1],
										strides=[1,2,2,1],
										padding="SAME")

		with tf.variable_scope("fc6"):
			# 在卷积和全连接之间转换要改变卷积计算的矩阵形状
			conv5_shape = conv5_pool.get_shape()
			column = conv5_shape[1]*conv5_shape[2]*conv5_shape[3]
			fc6_input = tf.reshape(conv5_pool, [self.batch, column])
			fc6_weight = tf.get_variable("fc6_weight",
										 shape=[column, 4096])
			fc6_biases = tf.get_variable("fc6_biases", shape=[4096],
										 dtype=tf.float32,initializer=tf.constant_initializer(0.01))
			fc6 = tf.nn.bias_add(tf.matmul(conv5_pool, fc6_weight), fc6_biases)

		with tf.variable_scope("fc7"):
			fc7_weight = tf.get_variable("fc7_weight",
										 shape=[4096, 4096])
			fc7_biases = tf.get_variable("f7_biases", shape=[4096],
										 dtype=tf.float32, initializer=tf.constant_initializer(0.01))
			fc7 = tf.nn.bias_add(tf.matmul(fc6, fc7_weight), fc7_biases)

		with tf.variable_scope("fc8"):
			fc8_weight = tf.get_variable("fc8_weight",
										 shape=[4096, 1000])
			fc8_biases = tf.get_variable("f7_biases", shape=[1000],
										 dtype=tf.float32, initializer=tf.constant_initializer(0.01))
			fc8 = tf.nn.bias_add(tf.matmul(fc6, fc7_weight), fc7_biases)

		return fc8