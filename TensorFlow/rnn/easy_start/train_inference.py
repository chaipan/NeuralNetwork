import tensorflow as tf
import numpy as np

FC1_NODE = 10
NUM_CLASSES = 5
FC2_NODE = NUM_CLASSES

STATE_COLS = 10
INPUT_COLS = 5



"""
没有计算出结果，工程作废
输入序列设置为[batches, series_length, input_data]三维矩阵，每次处理一个输入序列，循环整个batches
"""


def inference(batch_series):

	# 在一个循环体内分别构建四个层，顺序分别是 全连接1， tanh， 全连接2， soft+cross_entropy
	with tf.variable_scope("layer1_fc1"):
		state_init = tf.get_variable("state_tensor",
									 shape=[batch_series.shape[0], STATE_COLS],
									 initializer=tf.constant_initializer(0.0))
		fc1_weight = tf.get_variable("fc1_weight",
									 shape=[INPUT_COLS + STATE_COLS, FC1_NODE],
									 initializer=tf.truncated_normal_initializer)
		fc1_biases = tf.get_variable("fc1_biases",
									 shape=[FC1_NODE],
									 initializer=tf.constant_initializer(0.1))


	with tf.variable_scope("layer2_fc2"):
		fc2_weight = tf.get_variable("fc2_weight",
									 shape=[FC1_NODE, NUM_CLASSES],
									 initializer=tf.truncated_normal_initializer)
		fc2_biases = tf.get_variable("fc2_biases",
									 shape=[NUM_CLASSES],
									 initializer=tf.constant_initializer(0.1))


	logits = tf.constant(0, shape=[0,0],dtype=tf.float32)
	# 批处理， 每个batch处理一次，总共处理一个序列长度的次数
	for i in range(batch_series.shape[0]):
		batch_rows = batch_series[:, i, :]
		concated_tensor = tf.concat((batch_rows, state_init), axis=1)

		fc1 = tf.matmul(concated_tensor, fc1_weight) + fc1_biases
		state_later = tf.nn.tanh(fc1)

		fc2 = tf.matmul(state_later, fc2_weight) + fc2_biases

		logits = tf.concat([logits, fc2], axis=0)

		logits.append(fc2)
	logits = np.stack(logits)
	return np.stack(logits, axis=1)