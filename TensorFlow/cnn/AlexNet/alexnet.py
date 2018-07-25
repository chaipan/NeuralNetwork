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
										   initializer=tf.truncated_normal_initializer)
			conv1 = tf.nn.conv2d(self.X, filter=conv1_weight,
								 strides=[1,4,4,1],
								 padding="SAME")
			# 现在一般认为LRN没有效果，所以不用LRN
			conv1_relu1 = tf.nn.relu(tf.nn.bias_add(conv1_weight, conv1_biases))
			conv1_pooling = tf.nn.max_pool(conv1_relu1,
										   ksize=[])
