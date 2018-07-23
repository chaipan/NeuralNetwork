import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow.contrib.rnn as rnn


BATCH_SIZE = 64
SERIES_LENGTH = 28
INPUT_SIZE = 28

LSTM_NUMS = 64
NUM_CLASSES = 10

EPOCHES = 1000


def get_data():
	mnist = input_data.read_data_sets("D://PyCharm//NeuralNetwork//TensorFlow//data//mnist", one_hot=True)
	return mnist

def inference(input_tensor):
	lstm_cell = rnn.BasicLSTMCell(LSTM_NUMS,
								  forget_bias=0.8)
	lstm_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
	# rnn的输入值有两种形式，（batch_size, series_len, input_size）和[max_time, batch_size, depth]
	# 差别在于time_major的不同，If true用第二种形式，false用第一种形式，第一种的效率要高于第二中
	rnn_cell,(h_c, h_n) = tf.nn.dynamic_rnn(lstm_cell,
							inputs=input_tensor,
							dtype=tf.float32)
	out_put = tf.layers.dense(rnn_cell[:,-1,:], NUM_CLASSES)
	return out_put

def train(mnist):
	x = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, SERIES_LENGTH, INPUT_SIZE])
	y = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, NUM_CLASSES])

	logits = inference(x)
	accuracy = tf.metrics.accuracy(tf.argmax(y, 1), tf.argmax(logits, 1))[1]
	loss = tf.losses.softmax_cross_entropy(y, logits)

	train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

	init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())

	sess = tf.Session()
	sess.run(init_op)

	for step in range(EPOCHES):
		x_train, y_train = mnist.train.next_batch(BATCH_SIZE)
		x_train_shaped = np.reshape(x_train, [BATCH_SIZE, SERIES_LENGTH, INPUT_SIZE])
		_, loss_val, accuracy_val = sess.run([train_op, loss, accuracy],
											 feed_dict={x:x_train_shaped, y:y_train})
		print("step %d, accuracy %.4f"%(step, accuracy_val))


if __name__ == '__main__':
	mnist = get_data()
	train(mnist)